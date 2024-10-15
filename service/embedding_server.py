import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from config import embedding_model_path, embedding_server_url, embedding_server_port
app = FastAPI()
model = SentenceTransformer(embedding_model_path)


class Sentence(BaseModel):
    text: str


@app.get('/health_check')
def health_check():
    return 'Request OK'


@app.post('/embedding')
def get_embedding(sentence: Sentence):
    embedding = model.encode(sentence.text, normalize_embeddings=True).tolist()
    return {"text": sentence.text, "embedding": embedding}


if __name__ == '__main__':
    uvicorn.run(app, host=embedding_server_url, port=embedding_server_port)