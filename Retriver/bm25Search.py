from typing import List
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import TextNode
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import NodeWithScore

from jinja2 import Template
from ESHelper.estool import ESHelper
from config import dsl
from preprocess.llamainex_add_corpus import text_node_id_mapping


class bm25Search(BaseRetriever):
    def __init__(self, top_k):
        super().__init__()
        self.es_helper = ESHelper()
        self.top_k = top_k

    def _retrieve(self, query: QueryBundle) -> List[NodeWithScore]:
        if isinstance(query, str):
            query = QueryBundle(query)
        else:
            query = query

        result = []
        search_result = self.es_helper.search(query=query, top_k=self.top_k)

        if search_result['hits']['hits']:
            for record in search_result['hits']['hits']:
                text = record['_source']['context']
                node_with_score = NodeWithScore(node=TextNode(text=text,
                                                id_=text_node_id_mapping[text]),
                                                score=record['_score'])
                result.append(node_with_score)

        return result

if __name__ == '__main__':
    from pprint import pprint
    custom_bm25_retriever = bm25Search(top_k=3)
    query = "美日半导体协议是由哪两部门签署的？美日半导体协议是由美国商务部和日本经济产业省签署的。"
    t_result = custom_bm25_retriever.retrieve(str_or_query_bundle=query)
    pprint(t_result)

