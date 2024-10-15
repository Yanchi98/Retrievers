import json
import requests
from retry import retry
from config import embedding_server_url, embedding_server_port


@retry(exceptions=Exception, tries=3, max_delay=20)
def get_embedding(req_text: str):
    url = f'http://{embedding_server_url}:{embedding_server_port}/embedding'
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"text": req_text})
    new_req = requests.request("POST", url, headers=headers, data=payload)
    return new_req.json()['embedding']

