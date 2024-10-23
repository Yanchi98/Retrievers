import json
from ESHelper.estool import ESHelper
from config import corpus_path, es_index_name


es_helper = ESHelper()

with open(corpus_path, mode='r', encoding='utf-8') as f:
    content = json.loads(f.read())

corpus = content['corpus']
contents = [{'node_id': k, 'context': v} for k, v in corpus.items()]


es_helper.data_insert(contents=contents, index_name=es_index_name)


