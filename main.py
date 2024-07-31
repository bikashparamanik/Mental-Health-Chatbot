from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Load models
rag_model = pipeline("text-generation", model="distilgpt2")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load articles
with open('data/sample_articles.txt', 'r') as f:
    articles = f.read().split('\n')

# Precompute article embeddings
article_embeddings = sentence_model.encode(articles)

class RAGInput(BaseModel):
    prompt: str

class ClassificationInput(BaseModel):
    text: str

@app.get("/rag")
@app.post("/rag")
async def rag_endpoint(prompt: str):
    # Encode user input
    input_embedding = sentence_model.encode([prompt])
    
    # Calculate similarities
    similarities = cosine_similarity(input_embedding, article_embeddings)[0]
    
    # Get most similar article
    most_similar_idx = np.argmax(similarities)
    most_similar_article = articles[most_similar_idx]
    
    # Generate response using the LLM
    response = rag_model(f"Based on the following information: {most_similar_article}\n\nRespond to: {prompt}", max_length=200)[0]['generated_text']
    
    return {"response": response, "relevant_article": most_similar_article}

@app.get("/classification")
@app.post("/classification")
async def classification_endpoint(text: str):
    # Simple keyword-based classification (this is a placeholder and not accurate for real use)
    text = text.lower()
    if 'anxious' in text or 'worry' in text or 'nervous' in text:
        category = "Anxiety"
    elif 'sad' in text or 'hopeless' in text or 'depressed' in text or  'broken' in text or 'hurt' in text:
        category = "Depression"
    elif 'die' in text or 'suicide' in text or 'end my life' in text:
        category = "Suicidal"
    else:
        category = "Normal"
    
    return {"category": category}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)