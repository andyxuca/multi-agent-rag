import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def embed(texts):
    X = _embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False,
    )
    return np.asarray(X, dtype="float32")

def retrieve(query, k=3):
    index = faiss.read_index("index/faiss.index")
    meta = [json.loads(l) for l in open("index/meta.jsonl")]

    q = embed([query])

    faiss.normalize_L2(q)

    _, I = index.search(q, k)
    return [meta[i] for i in I[0]]
