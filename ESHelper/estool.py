import datetime
import time
import random
import json
import hashlib
from typing import List
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from jinja2 import Template
from util import setup_logger
from config import es_server_port, es_server_url, dsl, es_index_name


class ESHelper:

    def __init__(self):
        self.es = Elasticsearch(hosts=f'http://{es_server_url}:{es_server_port}')
        self.logger = setup_logger()
        self.create_index_dsl = Template(json.dumps(dsl['create_index']))
        self.search_dsl = Template(json.dumps(dsl['search']))

    def create_index(self, index_name:str) -> None:
        # 创建索引
        # res = es.index(index="doc_qa_dataset", id=1, document= create_inex)
        try:
            res = self.es.index(index=index_name, document= self.create_index_dsl)
            self.logger.info(f"Index created successfully: {res}")
        except Exception as e:
            self.logger.error(f"Error creating index: {e}")

    def search(self, query, top_k):
        search_result = self.es.search(index=es_index_name,
                                       body=self.search_dsl.render(query=query, top_k=top_k))

        return search_result

    def data_insert(self, contents: List[dict], index_name:str) -> None:
        success_cnt = 0
        start_time = time.time()
        def generate_actions(contents, index_name):
            for cnt in contents:
                action = {
                    "_index": index_name,
                    "_source": cnt,
                    "_id": hashlib.md5(cnt.get('context').encode('utf-8')).hexdigest()
                }
                yield action

        # 执行并行批量插入
        for success, info in parallel_bulk(self.es, generate_actions(contents, index_name),
                                           thread_count=4, chunk_size=1000):
            if not success:
                self.logger.error(f"Error inserting data: {info}")
            else:
                success_cnt += 1

        self.logger.info(f"Data successfully inserted: {success_cnt}/{len(contents)}, "
                         f"Time Cost: {time.time() - start_time},")