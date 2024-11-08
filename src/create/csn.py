import json
import argparse
import pandas as pd
from pathlib import Path 
from utils import save_tsv_dict, save_file_jsonl, NL2CodeDataset
import os
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets")

    args = parser.parse_args()
    
    commands = [
    "wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip",
    "unzip dataset.zip",
    "rm -r dataset.zip",
    "mv dataset CSN",
    "cd CSN && bash run.sh",
    "cd .."
    ]

    #[subprocess.run(command, shell=True, check=True) for command in commands]
    for lang in ['python', 'java', 'ruby', 'php', 'javascript', 'go']:
        path = Path(f'{args.output_dir}/csn_{lang}')
        path.mkdir(parents=True, exist_ok=True)
        qrels_path = Path(f'{path}/qrels')
        qrels_path.mkdir(parents=True, exist_ok=True)
        
        query_dataset = NL2CodeDataset(f'CSN/{lang}/test.jsonl', None)
        code_dataset = NL2CodeDataset(f'CSN/{lang}/codebase.jsonl', prefix = None)
        
        queries, docs, qrels = [], [], []
        url2id = {}
        i = 0
        for url, example in code_dataset.url2example.items():
            url2id[url] = i
            docs.append({'_id': f'{url2id[url]}_code', 'text': example['code'], 'title': example['title'], 'metadata': {}})
            i += 1
            
        for url, example in query_dataset.url2example.items():
            queries.append({'_id': f'{url2id[url]}_query', 'text': example['nl'], 'metadata': {}})
            qrels.append({"query-id": f'{url2id[url]}_query', "corpus-id": f'{url2id[url]}_code', "score": 1})
        
        
        
        save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
        save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))
        qrels_path = os.path.join(path, "qrels", "test.tsv")
        save_tsv_dict(qrels, qrels_path, ["query-id", "corpus-id", "score"])

    subprocess.run('rm -r CSN', shell=True, check=True)
    subprocess.run('rm -r _MACOSX', shell=True, check=True)
if __name__ == "__main__":
    main()