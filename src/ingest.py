import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def chunk_text(text, size=400):
    return [text[i:i+size] for i in range(0, len(text), size)]

def embed(texts):
    X = _embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=True,
    )
    return np.asarray(X, dtype="float32")

def main():
    os.makedirs("index", exist_ok=True)
    meta = []
    chunks = []

    with open("data/docs.jsonl") as f:
        for line in f:
            d = json.loads(line)
            for i, chunk in enumerate(chunk_text(d["text"])):
                meta.append({
                    "doc_id": d["doc_id"],
                    "chunk_id": i,
                    "text": chunk
                })
                chunks.append(chunk)

    # 🔹 use Qwen3 embeddings
    X = embed(chunks)

    index = faiss.IndexFlatIP(X.shape[1])
    faiss.normalize_L2(X)
    index.add(X)

    faiss.write_index(index, "index/faiss.index")
    with open("index/meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    print("Index built with Qwen3 embeddings.")

if __name__ == "__main__":
    main()
