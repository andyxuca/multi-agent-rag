from datasets import load_dataset
import json
import os

def main():
    os.makedirs("data", exist_ok=True)

    # load wikipedia dataset
    ds = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True, 
    )

    out = open("data/docs.jsonl", "w", encoding="utf-8")

    MAX_DOCS = 200  
    count = 0

    for row in ds:
        text = row["text"].strip()

        doc = {
            "doc_id": f"wiki_{row['title']}",
            "text": text
        }

        out.write(json.dumps(doc, ensure_ascii=False) + "\n")
        count += 1

        if count >= MAX_DOCS:
            break

    out.close()
    print(f"Wrote {count} Wikipedia documents.")

if __name__ == "__main__":
    main()
