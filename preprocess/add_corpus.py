import json
from config import corpus_path

with open(corpus_path, mode='r', encoding='utf-8') as f:
    content = json.loads(f.read())

corpus = content['corpus']
queries = list(content['queries'].values())
# query_relevant_docs: {query:node_id}
query_relevant_docs = {content['queries'][k]: v for k, v in content['relevant_docs'].items()}
node_id_text_mapping = content['corpus']
text_node_id_mapping = {v: k for k, v in node_id_text_mapping.items()}
