#!/usr/bin/env python3

import argparse
import csv
import os
import random
import string
import time
import torch
from typing import Any, List

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import BaseRetriever
from langchain.llms import GPT4All, LlamaCpp, OpenAI
from langchain import HuggingFacePipeline
from langchain.schema import Document
from langchain.callbacks.manager import Callbacks

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from constants import CHROMA_SETTINGS

from db import load_documents


def count_sentence_delimiters(text):
    sentence_delimiters = ['.', ',', ':', '?', '!', '\n']
    return sum([text.count(delimiter) for delimiter in sentence_delimiters])

class FilteredRetriever(BaseRetriever):
    def __init__(self, retriever, args):
        super().__init__()
        self.retriever = retriever
        self.args = args

    def get_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Retrieve documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        docs: List[Document] = self.retriever.get_relevant_documents(query, **kwargs)
        filtered_docs = list([doc for doc in docs 
                              if count_sentence_delimiters(doc.page_content) <= self.args.bib_delim_threshold and len(doc.page_content) >= self.args.min_doc_len])
        if len(filtered_docs) < self.args.target_source_chunks:
            print(f"Warning: only {len(filtered_docs)} documents found with less than {self.args.bib_delim_threshold} sentence delimiters.")
        
        return filtered_docs[:self.args.target_source_chunks]

    async def aget_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        raise Exception("Not implemented")

def summarization(llm, args):
    """
    Summarize a given text with a given model.
    """
    docs = load_documents(args.summary_documents)

    with open(f"{args.results}/summaries.csv", "w", newline='') as summaries_file:
        summaries_writer = csv.writer(summaries_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        summaries_writer.writerow(["title", "summary", "duration", "input", "file_path"])
        for doc in docs:
            title = doc.metadata.get("title", "")
            file_path = doc.metadata.get("file_path", "")
            print(f"Document: {doc.page_content}")
            if len(doc.page_content) > args.max_doc_len:
                # warn the user that the document is too long
                print(f"Warning: document {file_path} is too long ({len(doc.page_content)} characters). Only the first {args.max_doc_len} characters will be used.")

            input = f"### Human\n {args.prompt}\n\n ### Text \n{doc.page_content[:args.max_doc_len]} \n\n ### Assistant\n"
            start = time.time()
            summary = llm.predict(input)#, max_new_tokens=2048)
            end = time.time()
            duration = end - start

            print("Title:", title)
            print("The summary is:", summary)
            print("========================================\n\n")

            summaries_writer.writerow([title, summary, duration, input, file_path])

        summaries_writer.writerow(["experiment settings", str(args)])
 
def setup_qa(llm, args):
    embeddings = HuggingFaceEmbeddings(model_name=args.embeddings_model_name)
    db = Chroma(persist_directory=args.persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = FilteredRetriever(db.as_retriever(search_kwargs={"k": args.prefilter_target_source_chunks}), args)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    return qa

def question_answering(llm, args):
    """
    Question answering with a given model.
    """
    qa = setup_qa(llm, args)
    if args.questions_file is not None:
        with open(args.questions_file, newline='') as questions_file:
            questions_reader = csv.reader(questions_file, delimiter=',', quotechar='"')
            questions = list(questions_reader)[1:]

        with open(f"{args.results}/answers.csv", "w", newline='') as answers_file:
            answers_writer = csv.writer(answers_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            answers_writer.writerow(["question", "answer", "duration"] + [f"source_{i}" for i in range(args.target_source_chunks)])
            for n, query in questions:
                start = time.time()
                res = qa(query)
                end = time.time()
                duration = end - start
                answer, docs = res['result'], [] if args.hide_source else res['source_documents']
                print(f"\n\n> Question {n}:")
                print(query)
                print("\n> Answer:")
                print(answer)

                print("\n> Relevant sources:")
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
                    print(document.page_content)

                answers_writer.writerow([query, answer, duration] + [doc.page_content for doc in docs])
            
                print("\n========================================\n\n")

            answers_writer.writerow(["experiment settings", str(args)])

def repl(llm, args):
    """
    Read-eval-print loop for question answering with a given model.
    """
    args.mute_stream = False
    qa = setup_qa(llm, args)
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        print("Using the filtered archive.")
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        if args.print_archive:
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)

def setup_llm(args):
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    if args.model_type == "LlamaCpp":
        args.model_id = args.model_id if args.model_id is not None else ""
        llm = LlamaCpp(model_path=args.model_path, temperature=args.temperature, n_ctx=args.model_n_ctx, 
                       n_batch=args.model_n_batch, callbacks=callbacks, verbose=False)
    elif args.model_type == "GPT4All":
        args.model_id = args.model_id if args.model_id is not None else ""
        llm = GPT4All(model=args.model_path, temp=args.temperature, n_ctx=args.model_n_ctx, 
                      backend='gptj', n_batch=args.model_n_batch, callbacks=callbacks, verbose=False, n_threads=32)
    elif args.model_type == "huggingface":
        args.model_id = args.model_id if args.model_id is not None else "Salesforce/xgen-7b-8k-inst"
        if args.model_id == "Salesforce/xgen-7b-8k-inst":
            max_length = 8 * 1024
        else:
            max_length = 4 * 1024
        kwargs = {"max_length":max_length, "temperature":args.temperature, "trust_remote_code": True, "torch_dtype": torch.bfloat16}
        llm = HuggingFacePipeline.from_model_id(model_id=args.model_id,  task="text-generation", model_kwargs=kwargs, device=0)
    elif args.model_type == "openai":
        args.model_id = args.model_id if args.model_id is not None else "gpt-3.5-turbo"
        llm = OpenAI(model_name=args.model_id, temperature=args.temperature)
    else:
        raise ValueError(f"Model type {args.model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All, huggingface, openai.")

    return llm

def main():
    # Parse the command line arguments
    args = parse_arguments()
    if args.results is None:
        datestring = time.strftime("%Y%m%d-%H%M%S")
        datestring += "-" + ''.join(random.choice(string.ascii_lowercase) for i in range(4))
        args.results = f"results/{args.task}/{args.model_type}_{args.model_id.replace('/', '_')}_{args.target_source_chunks}_{datestring}"
    print(args)
    os.makedirs(args.results, exist_ok=True)
    llm = setup_llm(args)

    if args.task == "summarization":
        summarization(llm, args)
    elif args.task == "question_answering":
        question_answering(llm, args)
    elif args.task == "repl":
        repl(llm, args)
    else:
        raise ValueError(f"Task {args.task} is not supported. Please choose one of the following: summarize, question_answering, repl.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='kleioGPT: Explore different LLMs for academic question answering and summarization.')

    parser.add_argument("--task", "-T", type=str, default="repl")
    parser.add_argument("--results", "-R", type=str, default=None)

    # archive settings
    parser.add_argument("--summary-documents", "-S", type=str, default="summary_documents")
    parser.add_argument("--archive-directory", "-A", type=str, default="archive_documents")
    parser.add_argument("--persist-directory", "-D", type=str, default="db")
    parser.add_argument("--embeddings-model-name", "-E", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--bib-delim-threshold", type=int, default=25)
    parser.add_argument("--target-source-chunks", type=int, default=4)
    parser.add_argument("--prefilter-target-source-chunks", type=int, default=20)
    parser.add_argument("--min-doc-len", type=int, default=200)
    parser.add_argument("--print-archive", action='store_true', default=True)

    # LLM settings
    parser.add_argument("--model-type", "-M", type=str, default="GPT4All")
    parser.add_argument("--model-id", "-I", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="models/ggml-gpt4all-j-v1.3-groovy.bin")
    parser.add_argument("--model-n-ctx", "-C", type=int, default=1024)
    parser.add_argument("--model-n-batch", "-B", type=int, default=8)
    parser.add_argument("--questions-file", "-Q", type=str, default=None)
    parser.add_argument("--hide-source", "-H", action='store_true', default=False)
    parser.add_argument("--prompt", "-P", type=str, default="You are an assistant for academic research. Summarize the following text for an expert audience in 5 to 10 sentences:\n")
    parser.add_argument("--mute-stream", action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.', default=True)
    parser.add_argument("--max-doc-len", type=int, default=10000,
                        help='Maximum number of characters per document.')
    parser.add_argument("--temperature", "-t", type=float, default=0.3)

    return parser.parse_args()

if __name__ == "__main__":
    main()
