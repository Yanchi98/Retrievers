import json
import requests
from retry import retry


@retry(exceptions=Exception, tries=3, max_delay=20)
def get_bge_embedding(req_text: str):
    url = "http://localhost:50073/embedding"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"text": req_text})
    new_req = requests.request("POST", url, headers=headers, data=payload)
    return new_req.json()['embedding']

