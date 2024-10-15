import numpy as np
from faiss import IndexFlatIP
from util import setup_logger
from config import *

class VectorSearch:
    def __init__(self, top_k, faiss_index, query_rewrite=False)->None:
        self.logger = setup_logger()
        self.top_k = top_k
        self.faiss_index = faiss_index
        self.query_rewrite = query_rewrite
        corpus_embeddings = np.load(corpus_embedding_path)
        self.faiss_index.add(corpus_embeddings)
        self.logger.info("Corpus Embeddings loaded")



if __name__ == '__main__':
    faiss_index = IndexFlatIP(64)
    testdata = np.random.random((5, 64))
    faiss_index.add(testdata)
    searchdata = np.random.random((1, 64))
    faiss_index.search(searchdata, 2)