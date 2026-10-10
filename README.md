# Veritext — Document Intelligence & RAG

A full-stack Retrieval-Augmented Generation system: a FastAPI backend, a
persistent vector database, a local embedding model, and an LLM for
grounded, cited answer generation — paired with a premium web frontend.


## Features at a glance

- Upload several documents at once (PDF, TXT, MD); drag-and-drop or browse
- Page-aware PDF indexing: citations show the exact page ("report.pdf - page 3")
- Conversational follow-ups: recent turns are used for both search and answering
- Documents persist across server restarts (ChromaDB + `documents.json` registry)
- Chat history and a recent-questions list persist across browser refreshes
- Voice input, light/dark theme, conversation export, Insights dashboard

## Architecture

```
 ┌────────────┐      multipart/form-data       ┌──────────────────────┐
 │  Frontend  │ ─────────────────────────────▶ │   FastAPI  (app/)    │
 │ (browser)  │ ◀───────────────────────────── │                       │
 └────────────┘         JSON responses          │  1. ingestion.py      │
                                                 │     parse + chunk    │
                                                 │  2. vectorstore.py    │
                                                 │     ChromaDB +       │
                                                 │     sentence-        │
                                                 │     transformers     │
                                                 │  3. llm.py            │
                                                 │     Gemini: re-rank  │
                                                 │     + grounded answer│
                                                 └──────────────────────┘
                                                          │
                                                          ▼
                                                 ┌──────────────────────┐
                                                 │  ChromaDB (on disk)  │
                                                 │  ./chroma_db         │
                                                 └──────────────────────┘
```

**Pipeline, end to end:**

1. **Upload** (`POST /documents/upload`) — a `.txt`, `.md`, or `.pdf` file is
   parsed (`pypdf` for PDFs), cleaned, and split into overlapping,
   paragraph-aware chunks (`app/ingestion.py`).
2. **Embed + index** — each chunk is embedded locally with
   `sentence-transformers` (`all-MiniLM-L6-v2`, no external API call needed)
   and stored in a persistent **ChromaDB** collection with cosine similarity
   (`app/vectorstore.py`).
3. **Query** (`POST /query`) — the question is embedded and matched against
   the vector store to retrieve the most similar chunks.
4. **Hybrid re-ranking** (optional, default on) — the top candidates are
   handed to Gemini to re-rank by true relevance, combining sparse/dense
   vector retrieval with an LLM-based relevance judgment (`app/llm.py`).
5. **Grounded generation** — Gemini is prompted to answer **using only** the
   retrieved passages, citing them inline as `[1]`, `[2]`, etc., so every
   claim in the answer is traceable to a specific source chunk.

If no `GEMINI_API_KEY` is configured, steps 4–5 degrade gracefully: the
API still returns the retrieved passages (pure semantic search), just
without an LLM-written answer — useful for demonstrating the retrieval
engine on its own.

## Setup

**Quick start (macOS/Linux):**

```bash
cd stacks-backend
./run.sh
```

This creates a virtual environment, installs dependencies, copies
`.env.example` to `.env` if missing, and starts the server. Add your
`GEMINI_API_KEY` to `.env` before asking questions (retrieval still
works without it).

**Manual setup (any OS):**

```bash
cd stacks-backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GEMINI_API_KEY

uvicorn app.main:app --port 8000
```

**Note on `--reload`:** if you want the server to auto-restart when you edit
backend code, use `--reload --reload-dir app` instead of the plain command
above — restricting reload to just the `app/` folder stops it from also
watching `chroma_db/` (which the app itself writes to on every upload or
query) and restarting itself mid-request. For normal use or a demo, the
plain command above (no reload) is simplest and most stable.

The first run will download the `all-MiniLM-L6-v2` embedding model
(~80 MB) — this requires internet access once, then it's cached locally.

Open `frontend/index.html` directly in a browser. It's a full premium
UI — document index, chat, streaming-style pipeline states, and a
sources drawer with relevance bars and clickable citations — built on
the same design system as the standalone web app, but wired to real
HTTP calls against this backend instead of running client-side. It
talks to `http://localhost:8000` by default; change the API URL field
in the top bar if you deploy the backend elsewhere, then click Connect.

## Verifying everything works (no manual clicking required)

Once the server is running, in a second terminal:

```bash
python scripts/smoke_test.py
```

This uploads a test document, lists it, asks a question against it,
confirms an answer with sources comes back, then deletes it — exercising
every endpoint end to end. Run this before your demo/viva to confirm the
whole pipeline is healthy; every line should print `[PASS]`.

## API reference

| Method | Endpoint                  | Description                                   |
|--------|----------------------------|------------------------------------------------|
| GET    | `/health`                 | Health check + whether an LLM key is configured |
| POST   | `/documents/upload`       | Upload a document (multipart file)             |
| GET    | `/documents`              | List indexed documents with stats              |
| DELETE | `/documents/{doc_id}`     | Remove a document and its chunks               |
| POST   | `/query`                  | Ask a question; returns a grounded, cited answer |

Interactive API docs (Swagger UI) are auto-generated by FastAPI at
`http://localhost:8000/docs` once the server is running — useful for
demonstrating the API directly in a viva without even opening the frontend.

### Example: `POST /query`

```json
{
  "question": "What is the ARM load-store architecture?",
  "top_k": 5,
  "hybrid_rerank": true
}
```

```json
{
  "answer": "ARM uses a load-store architecture, meaning arithmetic and logic operations only act on registers [1]...",
  "sources": [
    { "rank": 1, "doc_id": "a1b2c3d4e5f6", "doc_name": "arm-notes.txt", "chunk_index": 2, "text": "...", "score": 0.83 }
  ],
  "used_llm": true
}
```

## Why this design (for your report / viva)

- **Real vector search, not a toy**: ChromaDB with cosine similarity over
  sentence-transformer embeddings is the same category of retrieval used in
  production RAG systems, just at a scale appropriate for a project.
- **Hybrid retrieval**: combines dense vector similarity (fast, scalable
  first pass) with an LLM re-ranking step (slower, more accurate second
  pass) — a well-known real-world RAG pattern for balancing speed and
  quality.
- **Grounded generation with citations**: the model is instructed to answer
  only from retrieved context and to cite every claim, which is the core
  idea that separates RAG from a plain chatbot — it reduces hallucination
  and makes answers auditable back to source text.
- **Graceful degradation**: the system is useful even without an LLM key
  (pure retrieval), which is a good resilience property to be able to
  explain and demonstrate.

## Extending it further

- Swap the in-memory `_documents` registry in `app/main.py` for a small
  SQLite table if you want document metadata to survive a server restart.
- Add authentication (`fastapi-users` or a simple API key header) before
  deploying this publicly.
- Deploy the API on Render/Railway/Fly.io and point `frontend/index.html`'s
  API URL field at the public address instead of localhost.
