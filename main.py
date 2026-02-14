import os
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Semantic Search with Re-ranking")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Dummy 100 product reviews
# Replace with real dataset
# -----------------------------
documents = [
    {"id": i, "content": f"This product review {i} talks about battery, performance and quality"}
    for i in range(100)
]

# -----------------------------
# Cache embeddings
# -----------------------------
doc_embeddings = []

def get_embedding(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Precompute embeddings at startup
@app.on_event("startup")
def embed_docs():
    global doc_embeddings
    if not doc_embeddings:
        texts = [doc["content"] for doc in documents]
        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        doc_embeddings = [np.array(e.embedding) for e in res.data]
        print("✅ Document embeddings cached")

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
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    try:
        score = float(resp.choices[0].message.content.strip())
    except:
        score = 5.0

    return score / 10  # normalize 0-1

# -----------------------------
# Search endpoint
# -----------------------------
@app.post("/")
def semantic_search(req: SearchRequest):
    start = time.time()

    # edge case empty query
    if not req.query.strip():
        return {
            "results": [],
            "reranked": False,
            "metrics": {"latency": 0, "totalDocs": len(documents)}
        }

    query_emb = get_embedding(req.query)

    # -------- Initial Retrieval --------
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
            "content": documents[idx]["content"],
            "score": float(max(0, min(1, score))),
            "metadata": {"source": "reviews"}
        })

    # -------- Re-ranking --------
    if req.rerank and candidates:
        for c in candidates:
            c["score"] = llm_score(req.query, c["content"])

        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:req.rerankK]
        reranked = True
    else:
        reranked = False
        candidates = candidates[:req.rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": candidates,
        "reranked": reranked,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
