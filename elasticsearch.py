from elasticsearch import Elasticsearch

es = Elasticsearch(hosts='http://127.0.0.1:9200')

doc = {
    "mappings": {
        "properties": {
            "node_id": {
                "type": "text"
            },
            "context": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            }
        }
    }
}

# 创建索引
res = es.index(index="doc_qa_dataset", id=1, document=doc)
print(res)
print(res['result'])