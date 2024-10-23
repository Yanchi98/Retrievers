import os
import json

project_path = os.path.abspath(os.path.dirname(__file__))
corpus_path = os.path.join(project_path, 'data/test', 'doc_qa_dataset.json')

# embedding相关配置
em_model = 'bge-large-zh-v1.5'
dimension = 1024
embedding_server_url = "0.0.0.0"
embedding_server_port = 50072
embedding_model_path =  os.path.join(project_path, 'model', em_model)
corpus_embedding_path = os.path.join(project_path, 'data/test', f'doc_qa_dataset_{em_model}.npy')

# es配置
es_server_url = "127.0.0.1"
es_server_port = 9200
es_index_name = "doc_qa_dataset"
es_dir = '/home/ES/elasticsearch-7.10.2/bin'
es_user = 'ES'
start_command = './elasticsearch'

# dsl
dsl_path = os.path.join(project_path, 'ESHelper', 'DSL.json')
with open(dsl_path, "r") as f:
    dsl = json.load(f)

# gradio
gradio_host = "127.0.0.1"
gradio_port = 6006