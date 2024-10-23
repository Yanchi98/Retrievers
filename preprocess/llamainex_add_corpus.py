"""
该文件用于向量检索，主要功能包括：
1. 从指定的 JSON 文件加载语料库和查询。
2. 生成查询相关文档的映射。
3. 创建节点 ID 和文本之间的映射关系。
"""

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
