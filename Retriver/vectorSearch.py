import json
import numpy as np
from typing import List
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import TextNode
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.schema import QueryType

from faiss import IndexFlatIP
from util import setup_logger
from config import *
from EmbeddingHelper.embedding_api import get_embedding

class VectorSearch(BaseRetriever):
    def __init__(self, top_k, faiss_index, query_rewrite=False)->None:
        super().__init__()
        self.logger = setup_logger()
        self.top_k = top_k
        self.faiss_index = faiss_index
        self.query_rewrite = query_rewrite

        corpus_embeddings = np.load(corpus_embedding_path)
        corpus_embeddings = np.array(corpus_embeddings, dtype=np.float32)
        self.faiss_index.add(corpus_embeddings)
        self.logger.info("Corpus Embeddings loaded")

        with open(corpus_path, mode='r', encoding='utf-8') as f:
            content = json.loads(f.read())
        corpus = content['corpus']
        self.text_node_id_mapping = {text:node_id for node_id, text in corpus.items()}
        self.corpus = list(corpus.values())

    def _retrieve(self, query:QueryType) -> List[NodeWithScore]:
        if isinstance(query, str):
            query = QueryBundle(query)

        result = []
        query_embedding = get_embedding(req_text=query.query_str)
        distances, doc_indices = self.faiss_index.search(np.array([query_embedding], dtype=np.float32), self.top_k)

        for i, sent_index in enumerate(doc_indices.tolist()[0]):
            text = self.corpus[sent_index]
            node_with_score = NodeWithScore(node=TextNode(text=text, id_=self.text_node_id_mapping[text]),
                                            score=distances.tolist()[0][i])
            result.append(node_with_score)

        return result


if __name__ == '__main__':
    faiss_index = IndexFlatIP(dimension)
    vector_search_retriever = VectorSearch(top_k=3, faiss_index=faiss_index)
    query = "美日半导体协议是由哪两部门签署的？美日半导体协议是由美国商务部和日本经济产业省签署的。"
    t_result = vector_search_retriever.retrieve(str_or_query_bundle=query)
    print(t_result)
    faiss_index.reset()