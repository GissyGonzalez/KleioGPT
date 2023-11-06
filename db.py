#!/usr/bin/env python3

import os
import glob
from typing import List, Optional, Mapping, Any, Iterator
from multiprocessing import Pool
from tqdm import tqdm
import shutil

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    #PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.document_loaders.pdf import BasePDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
import langchain.schema


from constants import CHROMA_SETTINGS

import argparse

# modified to load all documents at once
class PyMuPDFParser(BaseBlobParser):
    """Parse PDFs with PyMuPDF."""

    def __init__(self, text_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``fitz.Page.get_text()``.
        """
        self.text_kwargs = text_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import fitz

        with blob.as_bytes_io() as file_path:
            doc = fitz.open(file_path)  # open document

            page_contents = [(page.number, page.get_text(**self.text_kwargs))
                             for page in doc]

            yield langchain.schema.Document(
                page_content="\n".join([page_content for _, page_content in page_contents]),
                metadata=dict({
                            "source": blob.source,
                            "file_path": blob.source,
                            "total_pages": len(doc),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        }
                ),
            )

            # yield from [
            #     langchain.schema.Document(
            #         page_content=page.get_text(**self.text_kwargs),
            #         metadata=dict(
            #             {
            #                 "source": blob.source,
            #                 "file_path": blob.source,
            #                 "page": page.number,
            #                 "total_pages": len(doc),
            #             },
            #             **{
            #                 k: doc.metadata[k]
            #                 for k in doc.metadata
            #                 if type(doc.metadata[k]) in [str, int]
            #             },
            #         ),
            #     )
            #     for page in doc
            # ]

class PyMuPDFLoader(BasePDFLoader):
    """Loader that uses PyMuPDF to load PDF files."""

    def __init__(self, file_path: str) -> None:
        """Initialize with file path."""
        try:
            import fitz  # noqa:F401
        except ImportError:
            raise ImportError(
                "`PyMuPDF` package not found, please install it with "
                "`pip install pymupdf`"
            )

        super().__init__(file_path)

    def load(self, **kwargs: Optional[Any]) -> List[Document]:
        """Load file."""

        parser = PyMuPDFParser(text_kwargs=kwargs)
        blob = Blob.from_path(self.file_path)
        return parser.parse(blob)


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(archive_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(archive_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(args, ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {args.archive_directory}")
    documents = load_documents(args.archive_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {args.archive_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {args.chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main():
    args = parse_arguments()

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=args.embeddings_model_name)

    if args.task == "import":
        if does_vectorstore_exist(args.persist_directory):
            # Update and store locally vectorstore
            print(f"Appending to existing vectorstore at {args.persist_directory}")
            db = Chroma(persist_directory=args.persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
            collection = db.get()
            texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
            print(f"Creating embeddings. May take some minutes...")
            db.add_documents(texts)
        else:
            # Create and store locally vectorstore
            os.makedirs(args.persist_directory, exist_ok=True)
            print("Creating new vectorstore")
            texts = process_documents(args)
            print(f"Creating embeddings. May take some minutes...")
            db = Chroma.from_documents(texts, embeddings, persist_directory=args.persist_directory, client_settings=CHROMA_SETTINGS)
        db.persist()
        db = None

        print(f"Import complete! You can now run kleio.py to interact with documents")
    elif args.task == "delete":
        print(f"Deleting vectorstore at {args.persist_directory}")
        shutil.rmtree(args.persist_directory)
    else:
        raise ValueError(f"Unsupported task '{args.task}'")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Manage documents in a vectorstore.")

    parser.add_argument("--task", "-T", type=str, default="import", choices=["import", "delete"])
    parser.add_argument("--archive-directory", "-A", type=str, default="archive_documents")
    parser.add_argument("--persist-directory", "-D", type=str, default="db")
    parser.add_argument("--embeddings-model-name", "-E", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", "-C", type=int, default=500)
    parser.add_argument("--chunk-overlap", "-O", type=int, default=50)

    return parser.parse_args()


if __name__ == "__main__":
    main()
