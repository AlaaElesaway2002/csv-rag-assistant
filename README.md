# The Lot — CSV RAG Assistant

A Streamlit chat assistant that answers questions over one or more uploaded
CSVs using retrieval-augmented generation, grounded with visible source
citations.

## Pipeline

```
01_documents.py            CSV rows -> RAG documents
        |
02_preprocessing.py        text cleaning (lexical normalization)
        |
03_chunking.py              documents -> chunks
        |
04_vector_representation.py chunks -> TF-IDF / BM25 index + embeddings
        |
05_create_chroma_store.py   embeddings -> persistent ChromaDB vector store
        |
06_retrieve_context.py      query -> top-k chunks -> context package
        |
07_prompting.py             context -> prompt -> Ollama LLM answer
        |
streamlit_app.py            chat UI, filters, source cards, benchmark tab
```

Five retrieval methods are selectable in the sidebar: **TF-IDF**, **BM25**,
**Embeddings**, **Hybrid** (lexical + semantic), and **Chroma (Vector DB)**.

**A note on the numbered file names:** `01_documents.py` etc. are not valid
Python identifiers, so `import 01_documents` is a `SyntaxError`. Every module
that needs another numbered module loads it at runtime instead, with
`importlib.import_module("01_documents")` — this works because it resolves
the module by its string/file name rather than as a Python identifier.

## Installation

```bash
pip install -r requirements.txt
```

For the embeddings / hybrid / Chroma retrievers, also install and run
[Ollama](https://ollama.com) locally if you want generated answers rather
than the raw retrieved context:

```bash
ollama pull deepseek-r1:1.5b
ollama serve
```

## How to run

```bash
streamlit run streamlit_app.py
```

Upload a CSV from the **Settings** tab (or let it fall back to the bundled
`car_price_dataset.csv` demo data), then ask a question in the **Chat** tab.

## Example questions

- "What's the cheapest Audi currently in the lot?"
- "Is there a one-owner electric BMW X5 from 2018 or later?"
- "Show me Ford Focus listings from 2020–2023 with five previous owners."

## API keys / LLM provider

The app supports two LLM backends and picks one automatically at startup:

1. **OpenRouter** (deployment) — used when `OPENROUTER_API_KEY` is present
   in Streamlit secrets.
2. **Local Ollama** (local dev) — used as a fallback when no key is set.
   Configure host/model in the Settings tab.

No API key appears anywhere in the Python files, and no `.env` file is used.

### Local dev (Ollama)

```bash
ollama pull deepseek-r1:1.5b
ollama serve
```

### Deployment (OpenRouter via Streamlit secrets)

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` locally
(or paste the same TOML into Streamlit Cloud's **Manage app → Secrets**):

```toml
OPENROUTER_API_KEY = "your_openrouter_key_here"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
```

The default model is free on OpenRouter so the deployment works without
adding credits. Swap to `gpt-4o-mini` or any other OpenRouter model id
once the account has credits.

At startup the app runs the snippet the assignment PDF specifies:

```python
try:
    if not prompting_mod.OPENROUTER_API_KEY:
        prompting_mod.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    prompting_mod.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", prompting_mod.OPENROUTER_MODEL)
except Exception:
    pass
```

The Settings tab shows which provider is active.
