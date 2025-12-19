import json
import faiss
import numpy as np

def dummy_embed(texts):
    rng = np.random.default_rng(0)
    return rng.normal(size=(len(texts), 384)).astype("float32")

def retrieve(query, k=3):
    index = faiss.read_index("index/faiss.index")
    meta = [json.loads(l) for l in open("index/meta.jsonl")]

    q = dummy_embed([query])
    faiss.normalize_L2(q)
    _, I = index.search(q, k)

    return [meta[i] for i in I[0]]
