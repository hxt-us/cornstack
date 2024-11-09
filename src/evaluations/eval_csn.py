import os
import torch
import random
import logging
import argparse
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




def evaluate(lang):
    model = SentenceTransformer("cornstack/CodeRankEmbed", trust_remote_code= True).to('cuda').to(torch.bfloat16)
    
    corpus, queries, qrels = GenericDataLoader(
                data_folder=os.path.join("datasets/", f'csn_{lang}')
            ).load(split="test")
    

    query_examples = [(k, f'Represent this query for searching relevant code: {v}') for k, v in queries.items()]
    code_examples = [(k, v['text']) for k, v in corpus.items()]
        
    qs = [ex[1] for ex in query_examples]
    cs = [ex[1] for ex in code_examples]
    
    nl_vecs = model.encode(qs, show_progress_bar= True, batch_size= 64)
    code_vecs = model.encode(cs, show_progress_bar= True, batch_size= 64)


    scores = np.matmul(nl_vecs, code_vecs.T)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    print(f"nl_vecs_shape: {nl_vecs.shape} "
          f"\t code_vecs_shape: {code_vecs.shape} "
          f"\t score_matrix_shape: {scores.shape}")

    nl_ids = [ex[0] for ex in query_examples]
    code_ids = [ex[0] for ex in code_examples]


    ranks = []
    for url, sort_id in zip(nl_ids, sort_ids):
        rank = 0
        find = False
        for i, idx in enumerate(sort_id[:1000]):
            if find is False:
                rank += 1
            if code_ids[idx] == list(qrels[url].keys())[0]:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    mrr =  float(np.mean(ranks))
    
    return mrr


set_seed(42)
Path('results/csn').mkdir(parents=True, exist_ok=True)

for lang in ['python', 'java', 'ruby', 'php', 'javascript', 'go']:
    mrr = evaluate(lang)
    print(f'{lang} MRR', mrr)

    result_data = {
        "language": lang,
    }

    with open(f"results/csn/overall_results.jsonl", 'a') as f:
        f.write(json.dumps({**result_data, **{'mrr': mrr}}) + "\n")



