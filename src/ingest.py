import json
import os
import numpy as np
import faiss

def chunk_text(text, size=400):
    return [text[i:i+size] for i in range(0, len(text), size)]

def dummy_embed(texts):
    # Placeholder — replace later with real embeddings
    rng = np.random.default_rng(0)
    return rng.normal(size=(len(texts), 384)).astype("float32")

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

    X = dummy_embed(chunks)
    index = faiss.IndexFlatIP(X.shape[1])
    faiss.normalize_L2(X)
    index.add(X)

    faiss.write_index(index, "index/faiss.index")
    with open("index/meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    print("Index built.")

if __name__ == "__main__":
    main()
