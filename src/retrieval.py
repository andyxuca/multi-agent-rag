import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)

_index = None
_meta = None

def embed(texts):
    X = _embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False,
    )
    return np.asarray(X, dtype="float32")

def _load_once():
    global _index, _meta
    if _index is None:
        _index = faiss.read_index("index/faiss.index")
    if _meta is None:
        with open("index/meta.jsonl") as f:
            _meta = [json.loads(l) for l in f]

def retrieve(query, k=3):
    _load_once()
    q = embed([query])
    faiss.normalize_L2(q)
    _, I = _index.search(q, k)
    return [_meta[i] for i in I[0]]
