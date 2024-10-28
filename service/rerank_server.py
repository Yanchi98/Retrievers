import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import reranker_model_path, reranker_server_url, reranker_server_port
app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)
model = AutoModelForSequenceClassification.from_pretrained(reranker_model_path)


class Pairs(BaseModel):
    pairs: List[list]

@app.get('/health_check')
def health_check():
    return 'Request OK'


@app.post('/rerank')
def rerank(pairs: Pairs):
    with torch.no_grad():
        inputs = tokenizer(pairs.pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

    rank_result = [{"cnt": pair[1], "score": score} for pair, score in zip(pairs.pairs, scores)]
    sort_rerank_result = sorted(rank_result, key=lambda k: k['score'], reverse=True)
    return sort_rerank_result


if __name__ == '__main__':
    uvicorn.run(app, host=reranker_server_url, port=reranker_server_port)