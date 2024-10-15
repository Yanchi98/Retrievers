import numpy as np
import json
from tqdm import tqdm
from config import corpus_path, dimension, corpus_embedding_path
from EmbeddingHelper.embedding_api import get_embedding

def build_with_context(context_type: str):
    with open(corpus_path, "r", encoding="utf-8") as f:
        content = json.loads(f.read())
    queries = list(content[context_type].values())
    query_num = len(queries)
    embedding_data = np.empty(shape=[query_num, dimension])

    for i in tqdm(range(query_num), desc="generate embedding"):
        embedding_data[i] = get_embedding(queries[i])

    np.save(corpus_embedding_path, embedding_data)


if __name__ == '__main__':
    build_with_context("corpus")