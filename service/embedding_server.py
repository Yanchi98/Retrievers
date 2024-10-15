import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from config import embedding_model_path
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
    uvicorn.run(app, host='0.0.0.0', port=50072)