# Semantic Search API with LLM Re-ranking

FastAPI-based semantic search system that retrieves relevant documents using embeddings and improves ranking using an LLM.

## Features
- Semantic search with OpenAI embeddings  
- Cosine similarity retrieval  
- Optional GPT re-ranking  
- FastAPI REST API  
- Cached embeddings for speed  
- Latency metrics  

## Tech Stack
FastAPI • OpenAI API • NumPy • Pydantic • Uvicorn

## Setup

Install dependencies:
pip install fastapi uvicorn openai numpy pydantic

Set API key:
export OPENAI_API_KEY=your_key   (Mac/Linux)  
set OPENAI_API_KEY=your_key      (Windows)

Run server:
uvicorn main:app --reload

## API Usage

GET /
Health check

POST /
Request:
{
  "query": "battery performance",
  "k": 10,
  "rerank": true,
  "rerankK": 5
}

Returns top relevant documents with scores and latency.

## Notes
- Embeddings generated once and cached  
- Vector search done before LLM to reduce cost  
- Designed to demonstrate modern AI retrieval systems  
