# Multi-Agent RAG

A multi-agent retrieval-augmented generation pipeline consistenting of a planner, retriever, and answerer. Uses the DeepSeek R1 reasoning model.

## Quick Start

1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Set env vars in a `.env` file:
```bash
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=your_base_url
DEEPSEEK_MODEL=your_model
```

3) Build the corpus and index:
```bash
python src/build_wiki_corpus.py
python src/ingest.py
```

4) Ask a question:
```bash
python src/app.py
```