import os

project_path = os.path.abspath(os.path.dirname(__file__))
corpus_path = os.path.join(project_path, 'data/test', 'doc_qa_dataset.json')
em_model = 'bge-large-zh-v1.5'
embedding_model_path =  os.path.join(project_path, 'model', em_model)
corpus_embedding_path = os.path.join(project_path, 'data/test', f'doc_qa_dataset_{em_model}.npy')
dimension = 1024
embedding_server_url = "127.0.0.1"
embedding_server_port = 6006