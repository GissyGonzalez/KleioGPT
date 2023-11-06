# KleioGPT

This project aims at making large language models such as ChatGPT accessible to historical research. It was originally based on [privateGPT](https://github.com/imartinez/privateGPT). It is the accompanying code for [If the Sources Could Talk: Evaluating Large Language Models for Research Assistance in History](https://arxiv.org/abs/2310.10808).

## Database

The script `db.py` allows to import an archive from a directory (defaulting to: `archive_documents`) as follows:

```
python3 db.py -T import --archive-directory DIRECTORY
```

This process might take some time depending on the size of the archive.

## Experiments

Provided you have a function to `submit` (run) a job in your infrastructure, you can run the following loops to reproduce the results reported in the paper.

```python
preamble = "You are helpful assistant talking to a curious academic expert human."
prompts = ["Summarize the following text in five sentences, including stating what is its main idea:"]
chunk_sizes = [0, 4, 8]

for model, model_id, model_args in [("huggingface", "stabilityai/StableBeluga-7B", "--max-doc-len=5000"), ("huggingface", "tiiuae/falcon-7b-instruct", "--max-doc-len=5000"), ("huggingface", "Salesforce/xgen-7b-8k-inst", "--max-doc-len=10000"), ("openai", "gpt-3.5-turbo", "--max-doc-len=5000")]:
    for task, args in list([("question_answering", f"-Q scratch/queries2.csv --target-source-chunks {c}") for c in chunk_sizes]) + list([("summarization", f"-P \"{preamble} {p}\"") for p in prompts]):
        command = f"python3 kleio.py -M {model} --model-id {model_id} -T {task} {args} --temperature 0.00001"
        submit(command, jobname=f'{model}-{model_id}-{task}')
```

## Copyright 2023 Giselle Gonzalez Garcia, Christian Weilbach

Licensed under Apache 2.0, same as privateGPT.
