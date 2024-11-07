
# 🌽 CORNSTACK: A HIGH-QUALITY CONTRASTIVE DATASET FOR BETTER CODE RETRIEVAL AND RERANKING

<p align="left">
    ℹ️&nbsp;<a href="#-about">About</a>
    | 📖&nbsp;<a href="#-more-about-cornstack">More About CORNSTACK</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 👀&nbsp;<a href="#-running-evaluation">Running Evaluation</a>
</p>



## ℹ️ About
* 🌽 **CORNSTACK** is a large-scale high-quality (text, code) pairs dataset for training and fine-tuning embedding models and re-rankers for code retrieval via contrastive learning. 
* We train **CodeRankEmbed**, a 137M bi-encoder, on 🌽 **CORNSTACK** and demonstrate considerably higher performance on a variety of code retrieval benchmarks, with substantial gains over current state-of-the-art code embedding models.
* By leveraging 🌽 **CORNSTACK**, we are the first to finetune LLMs as code rerankers. **CodeRankLLM**, our 7B code reranker, considerably improves performance over the retriever.


## 📖 More About CORNSTACK

The performance of code embedding models is highly contingent on the quality of the large-scale data used for contrastive training. Effective contrastive training hinges on satisfying two primary conditions: 
1) The positives are highly relevant to the query and not noisy
2) The negatives are semantically similar to the positives but do not directly address the query, a.k.a hard negatives.

Existing approaches heuristically source contrastive examples from large-scale open-source code data with limited filtering and mining, retaining irrelevant or incorrectly labeled <query, positive> pairs, which impair the models’ ability to learn robust and accurate representations. To address these challenges, we introduce curriculum-based hard negative mining and consistency filtering techniques and apply these techniques on the de-duplicated version of The Stack v2. More details on these specific curation techniques and how we use them to train embedding models and re-rankers in our paper coming soon!

## 🚀 Quick Start

Install the required dependencies:
```
pip install -r requirements.txt
```


## 👀 Running Evaluation

To reproduce the performance of **CodeRankEmbed** on popular code retrieval benchmarks, run the following commands: 

### COIR Evaluation
```
bash evaluation/eval_coir.py
```

### CSN Evaluation

```
bash evaluation/eval_csn.py
```



