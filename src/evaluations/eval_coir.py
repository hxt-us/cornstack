import coir
from coir.evaluation import COIR
from sentence_transformers import SentenceTransformer
from beir.retrieval import models
import torch

contrast_encoder = models.SentenceBERT()
contrast_encoder.q_model = SentenceTransformer("nomic-uiuc/CodeEmbed", trust_remote_code= True)
contrast_encoder.doc_model = SentenceTransformer("nomic-uiuc/CodeEmbed", trust_remote_code= True)
contrast_encoder.q_model.max_seq_length = 512
contrast_encoder.doc_model.max_seq_length = 512


tasks = coir.get_tasks(tasks= ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt",
                                      "codefeedback-st","codetrans-contest","synthetic-text2sql",
                                      "cosqa","codesearchnet","codesearchnet-ccr"])

# Initialize evaluation
evaluation = COIR(tasks=tasks,batch_size=256)

# Run evaluation
results = evaluation.run(contrast_encoder, output_folder=f"results/coir")
print(results)