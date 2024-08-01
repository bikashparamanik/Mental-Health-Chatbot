from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = FastAPI()

# Load models
rag_model = pipeline("text-generation", model="distilgpt2")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load articles
with open('data/sample_articles.txt', 'r') as f:
    articles = f.read().split('\n')

# Precompute article embeddings
article_embeddings = sentence_model.encode(articles)

# Load and prepare the mental health dataset
df = pd.read_csv('mental_health.csv')
# Remove rows with NaN values
df = df.dropna(subset=['statement', 'status'])
X = df['statement']
y = df['status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classification model
classification_model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])
classification_model.fit(X_train, y_train)

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
    # Use the trained model to predict the category
    category = classification_model.predict([text])[0]
    return {"category": category}

@app.get("/model_accuracy")
async def model_accuracy():
    accuracy = classification_model.score(X_test, y_test)
    return {"accuracy": accuracy}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)