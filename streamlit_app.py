"""
streamlit_app.py
-----------------
"The Lot" — a chat assistant over one or more uploaded CSVs, built on the
Lab 8 / Lab 9 RAG methodology: CSV rows -> documents -> chunks -> retriever ->
context package -> prompt -> LLM answer. Every answer shows the sources it
was grounded in.

v2: tabbed layout (Chat / Browse / Benchmark / Settings), live structured
filters shared by Chat + Browse, inline citations + a visible source grid,
streaming answers from Ollama, a caching fix, and (v3) a ChromaDB-backed
"Chroma (Vector DB)" retrieval option alongside TF-IDF / BM25 / Embeddings /
Hybrid.

Run with:
    streamlit run streamlit_app.py
"""

import hashlib
import html
import importlib
import io
import json
import os

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Pipeline modules. File names are numbered (01_documents.py, etc.) per the
# assignment's required structure. Numbered file names are NOT valid Python
# identifiers, so `import 01_documents` would be a SyntaxError -- we load
# them by string name with importlib instead, which works fine at runtime.
# --------------------------------------------------------------------------- #
documents_mod = importlib.import_module("01_documents")
preprocessing_mod = importlib.import_module("02_preprocessing")
chunking_mod = importlib.import_module("03_chunking")
vector_mod = importlib.import_module("04_vector_representation")
chroma_mod = importlib.import_module("05_create_chroma_store")
retrieve_mod = importlib.import_module("06_retrieve_context")
prompting_mod = importlib.import_module("07_prompting")
aggregation_mod = importlib.import_module("08_aggregation")

# Per the assignment's PDF: when deployed on Streamlit Cloud, the OpenRouter
# API key lives in st.secrets (TOML) and is read into prompting_mod at startup.
# Local dev falls back to local Ollama when no key is present.
try:
    if not prompting_mod.OPENROUTER_API_KEY:
        prompting_mod.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    prompting_mod.OPENROUTER_MODEL = st.secrets.get(
        "OPENROUTER_MODEL", prompting_mod.OPENROUTER_MODEL
    )
except Exception:
    pass

# --------------------------------------------------------------------------- #
# Page config + design system
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="The Lot — CSV RAG Assistant",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #F6F8FB;
    --panel: #FFFFFF;
    --panel-2: #F1F5F9;
    --line: #E5E7EB;
    --ink: #1F2937;
    --ink-dim: #6B7280;
    --blue: #2563EB;
    --blue-bright: #3B82F6;
    --blue-tint: #EFF4FF;
    --current: #16A34A;
    --outdated: #3B82F6;
}

html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Space Grotesk', sans-serif;
}
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; color: var(--ink) !important; }

/* ---- Hero ---- */

.lot-icon {
    flex-shrink: 0; width: 42px; height: 42px; border-radius: 10px;
    background: var(--blue-tint); display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
}


/* ---- Sidebar section headers ---- */
.side-h {
    display: flex; align-items: center; gap: 8px; margin: 18px 0 8px 0;
    font-weight: 600; font-size: 0.92rem; color: var(--ink);
}
.side-h .num {
    width: 20px; height: 20px; border-radius: 50%; background: var(--blue); color: #fff;
    font-size: 0.68rem; font-family: 'IBM Plex Mono', monospace; display: flex;
    align-items: center; justify-content: center; flex-shrink: 0;
}

/* ---- File chip ---- */
.file-chip {
    display: flex; align-items: center; gap: 10px; border: 1px solid var(--line);
    background: var(--panel-2); border-radius: 8px; padding: 8px 12px; margin-bottom: 6px;
}
.file-chip-icon { color: var(--current); font-size: 1rem; }
.file-chip-name { font-weight: 600; font-size: 0.82rem; color: var(--ink); }
.file-chip-meta { font-size: 0.7rem; color: var(--ink-dim); font-family: 'IBM Plex Mono', monospace; }

/* ---- Source card (grid) ---- */
.source-card {
    display: flex; flex-direction: column; gap: 6px; border: 1px solid var(--line);
    background: var(--panel); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04); height: 100%;
}
.source-icon-row { display: flex; align-items: center; justify-content: space-between; }
.source-icon { font-size: 1rem; }
.source-label { font-size: 0.66rem; color: var(--blue); font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.source-title { font-weight: 600; font-size: 0.85rem; color: var(--ink); }
.source-tag { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
.tag-current { color: var(--current); }
.tag-outdated { color: var(--outdated); }
.source-price { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.9rem; color: var(--ink); }
.source-date, .source-detail { font-size: 0.68rem; color: var(--ink-dim); font-family: 'IBM Plex Mono', monospace; }

/* ---- Citation markers (self-generated only -- never wraps raw LLM text) ---- */
.cite-sup { color: var(--blue); font-weight: 700; font-size: 0.72em; }

/* ---- Panels ---- */
.panel { border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04); }
.panel-label { font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.66rem; color: var(--blue); margin-bottom: 8px; font-weight: 600; }
.context-block { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; white-space: pre-wrap; background: var(--panel-2); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; color: var(--ink); line-height: 1.5; }

/* ---- Sidebar ---- */

section[data-testid="stSidebar"]{
    background:#F8FAFC !important;
    padding-top:18px;
}
.sidebar-card{ background:white;

    padding:18px;

    border-radius:14px;

    border:1px solid #E5E7EB;

    margin-bottom:18px;

    box-shadow:0 2px 8px rgba(0,0,0,.04);

}
.sidebar-title{ font-size:12px;

    font-weight:700;

    letter-spacing:2px;

    text-transform:uppercase;

    color:#2563EB;

    margin-bottom:14px;

}
/* ---- Buttons ---- */
.stButton>button { background-color: var(--blue) !important; color: #FFFFFF !important; border: none !important; font-weight: 600 !important; border-radius: 6px !important; font-family: 'Space Grotesk', sans-serif !important; }
.stButton>button:hover { background-color: var(--blue-bright) !important; }
button[kind="secondary"] { color: var(--outdated) !important; }

hr { border-color: var(--line) !important; }
/* Slider */


/* Radio button */


/* Toggle */
.stCheckbox input:checked + div,
.stToggle input:checked + div {
    background-color: #2563EB !important;
}

/* Buttons */
.stButton > button {
    background: #2563EB !important;
    color: white !important;
}
.app-header{
    background:white;
    border:1px solid #E5E7EB;
    border-radius:20px;
    padding:36px;
    margin-bottom:24px;
    box-shadow:0 10px 30px rgba(15,23,42,.08);
}

.header-left{
display:flex;
gap:22px;
align-items:center;
}

.header-icon{
    width:80px;
    height:80px;
    border-radius:20px;
    background:linear-gradient(135deg,#DBEAFE,#EEF2FF);
    display:flex;
    justify-content:center;
    align-items:center;
    font-size:42px;
}

.header-small{
font-size:12px;
letter-spacing:2px;
text-transform:uppercase;
color:#2563EB;
font-weight:600;
margin-bottom:6px;
}

.header-title{
font-size:34px;
font-weight:700;
color:#111827;
}

.header-desc{
margin-top:8px;
font-size:16px;
color:#6B7280;
max-width:760px;
line-height:1.6;
}
/* Radio Buttons */

.stRadio > div{
    gap:10px;
}

.stRadio label{
    background:#FFFFFF;
    border:1px solid #E5E7EB;
    border-radius:10px;
    padding:10px 14px;
    transition:.2s;
}

.stRadio label:hover{
    border-color:#2563EB;
    background:#EFF6FF;
}
/* تحويل المتغيرات اللونية الأساسية داخل صندوق السلايدر فقط إلى الرمادي */
div[data-testid="stSlider"] {
    --primary-color: #6B7280 !important;
    --thumb-color: #4B5563 !important;
    --track-color: #E5E7EB !important;
}

/* صبغ الخط النشط للمؤشر بالرمادي وتخطي لون النظام الأحمر */
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
    background: #6B7280 !important;
    background-color: #6B7280 !important;
}

/* صبغ المقبض الدائري بالرمادي الداكن */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #4B5563 !important;
    border-color: #4B5563 !important;
    box-shadow: none !important;
}

/* تحويل أرقام المؤشر (مثل 3 و 0.60) من الأحمر إلى الرمادي الداكن */
div[data-testid="stSlider"] div[data-styled-engine="true"],
div[data-testid="stSlider"] span,
div[data-testid="stSlider"] p,
div[data-testid="stSlider"] div {
    color: #4B5563 !important;
}

/* صيانة وتأكيد: إبقاء أزرار التطبيق باللون الأزرق المعتاد */
.stButton > button {
    background-color: #2563EB !important;
    color: white !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🤖"


# --------------------------------------------------------------------------- #
# Cached pipeline construction
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def _build_corpus(files: tuple):
    """files: tuple of (name, bytes). Plain DataFrame outputs -> cache_data is
    the right tool here (unlike the vectorizer/bm25 caches below)."""
    doc_frames = []
    for name, content in files:
        raw_df = pd.read_csv(io.BytesIO(content))
        doc_frames.append(documents_mod.build_documents_any(raw_df, name))
    documents_df = documents_mod.combine_documents(doc_frames)
    chunks_df = chunking_mod.build_chunks(documents_df)
    return documents_df, chunks_df


@st.cache_resource(show_spinner=False)
def _load_tfidf(_chunks_df: pd.DataFrame, cache_key: str):
    """cache_resource, not cache_data: a fitted TfidfVectorizer + sparse
    matrix are stateful objects, not plain serializable data. cache_data
    would re-copy them on every access; cache_resource stores by reference."""
    return vector_mod.build_tfidf(_chunks_df)


@st.cache_resource(show_spinner=False)
def _load_bm25(_chunks_df: pd.DataFrame, cache_key: str):
    return vector_mod.build_bm25(_chunks_df)


@st.cache_resource(show_spinner=False)
def _get_filtered_indexes(file_sig: str, filter_sig: str, _full_chunks_df: pd.DataFrame, _filtered_document_ids):
    """One cached bundle per (file, filter) combination: filtered chunk table
    + a lexical index rebuilt on just that subset (cheap at 10k rows). Empty
    filter (filter_sig == 'ALL') reuses the full-corpus chunk table untouched
    so the very common no-filter case is just as cheap as before. Embeddings
    are deliberately NOT rebuilt here -- see _embeddings_for_chunks below,
    which index-subsets the already-computed full-corpus array instead."""
    if _filtered_document_ids is None:
        filtered_chunks_df = _full_chunks_df
    else:
        mask = _full_chunks_df["document_id"].isin(_filtered_document_ids)
        filtered_chunks_df = _full_chunks_df[mask].reset_index(drop=True)
    vectorizer, matrix = vector_mod.build_tfidf(filtered_chunks_df)
    bm25 = vector_mod.build_bm25(filtered_chunks_df)
    return filtered_chunks_df, vectorizer, matrix, bm25


@st.cache_resource(show_spinner=False)
def _load_embedding_model():
    return vector_mod.load_embedding_model()


@st.cache_data(show_spinner=False)
def _load_embeddings(_model, chunks_df: pd.DataFrame, _model_key: str):
    """Full-corpus only, never keyed on filter_sig -- this is the expensive
    step filters must not trigger again. Embeddings for a filtered subset are
    obtained by index-subsetting this array (see _embeddings_for_chunks)."""
    return vector_mod.embed_chunks(_model, chunks_df)


@st.cache_resource(show_spinner=False)
def _load_chroma_collection(_chunks_df: pd.DataFrame, _chunk_embeddings, cache_key: str):
    """Full-corpus only, same reasoning as _load_embeddings above: this is
    the expensive step (writing every row into the persistent Chroma store),
    so it's keyed on file_sig, not filter_sig. Filtering for the Chroma
    retriever happens client-side after querying (see retrieve_top_k_chroma).

    The cache_key (file_sig) is also folded into the collection name so a
    different CSV can't reuse a stale collection left on disk from a previous
    run -- chromadb.PersistentClient.get_or_create_collection would otherwise
    happily return whatever was already persisted under the default name."""
    collection = chroma_mod.create_chroma_collection(
        _chunks_df,
        collection_name=f"rag_chunks_{cache_key[:8]}",
    )
    chroma_mod.add_chunks_to_chroma(collection, _chunks_df, _chunk_embeddings)
    return collection


def _embeddings_for_chunks(filtered_chunks_df: pd.DataFrame, full_chunk_embeddings):
    """chunk_id is assigned sequentially in the same order chunk_embeddings
    was computed in, so it doubles as a row index into that array -- no
    re-encoding needed for a filtered subset, just a numpy fancy-index."""
    if full_chunk_embeddings is None:
        return None
    positions = filtered_chunks_df["chunk_id"].to_numpy()
    return full_chunk_embeddings[positions]


def _file_signature(files):
    h = hashlib.sha256()
    for name, content in files:
        h.update(name.encode())
        h.update(content)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Header + tabs (Settings tab holds the file uploader -- it has to be
# declared here, early, because Streamlit widgets return their value at the
# point they're declared in code, and everything below needs `files`).
# --------------------------------------------------------------------------- #

st.markdown("""
<div class="app-header">

<div class="header-left">

<div class="header-icon">
🧠
</div>

<div>

<div class="header-small">
Grounded Retrieval Augmented Generation
</div>

<div class="header-title">
CSV RAG Assistant
</div>

<div class="header-desc">
Upload one or more CSV files, ask questions in natural language,
retrieve the most relevant records, and generate grounded answers using your local LLM.
</div>

</div>

</div>

</div>
""", unsafe_allow_html=True)

tab_chat, tab_browse, tab_benchmark, tab_settings = st.tabs(
    ["💬 Chat", "🔎 Browse", "📊 Benchmark", "⚙️ Settings"]
)

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "car_price_dataset.csv")

with tab_settings:
    st.markdown("### Data source")
    uploaded_files = st.file_uploader(
        "Upload CSV files", type="csv", accept_multiple_files=True,
        help="Each row becomes a document. The car-listing dataset gets a specialised, "
             "human-readable rendering; any other CSV falls back to a generic 'field: value' rendering.",
    )
    files = [(f.name, f.getvalue()) for f in uploaded_files] if uploaded_files else []
    if not files and os.path.exists(DEFAULT_CSV):
        with open(DEFAULT_CSV, "rb") as f:
            files = [("car_price_dataset.csv", f.read())]
        st.caption("No files uploaded — using the bundled `car_price_dataset.csv` as a demo corpus.")

    st.markdown("""
<div class="sidebar-card">
<div class="sidebar-title">
🤖 AI Model
</div>
""", unsafe_allow_html=True)
    if prompting_mod.OPENROUTER_API_KEY:
        st.success(f"Using OpenRouter — model: `{prompting_mod.OPENROUTER_MODEL}`")
        st.caption("Key read from `st.secrets`. Ollama fields below are unused while the key is present.")
    else:
        st.info("No `OPENROUTER_API_KEY` in st.secrets — using local Ollama below.")
    ollama_host = st.text_input("Host", value=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ollama_model = st.text_input("Model", value=os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b"))
    ollama_ok, ollama_info = prompting_mod.ollama_status(ollama_host)
    if prompting_mod.OPENROUTER_API_KEY:
        or_ok, or_info = prompting_mod.openrouter_status(prompting_mod.OPENROUTER_API_KEY)
        if or_ok:
            st.caption(f"OpenRouter reachable ({len(or_info)} models available).")
        else:
            st.warning(f"OpenRouter not reachable: {or_info}")
    elif ollama_ok:
        st.success(f"Ollama connected. Models: {', '.join(ollama_info) if ollama_info else '(none pulled yet)'}")
    else:
        st.caption(f"Ollama not reachable ({ollama_info}). Answers will show the built prompt instead.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("### Session")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear chat history", width="stretch"):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("Clear cache", width="stretch"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

if not files:
    st.info("Upload at least one CSV from the **Settings** tab to get started.")
    st.stop()

# --------------------------------------------------------------------------- #
# Build corpus + indexes
# --------------------------------------------------------------------------- #

with st.spinner("Reading rows and building documents..."):
    documents_df, chunks_df = _build_corpus(tuple(files))

file_sig = _file_signature(files)

with st.spinner("Indexing (TF-IDF + BM25)..."):
    tfidf_vectorizer, tfidf_matrix = _load_tfidf(chunks_df, file_sig)
    bm25 = _load_bm25(chunks_df, file_sig)

embedding_model, embedding_error = _load_embedding_model()
chunk_embeddings = None
if embedding_model is not None:
    with st.spinner("Embedding rows (first run only)..."):
        chunk_embeddings = _load_embeddings(embedding_model, chunks_df, "all-MiniLM-L6-v2")
embeddings_available = embedding_model is not None and chunk_embeddings is not None

chroma_collection = None
chroma_error = None
if embeddings_available:
    try:
        with st.spinner("Writing rows into the Chroma vector store (first run only)..."):
            chroma_collection = _load_chroma_collection(chunks_df, chunk_embeddings, file_sig)
    except Exception as exc:  # noqa: BLE001 -- Chroma is additive, never fatal to the app
        chroma_error = str(exc)
chroma_available = chroma_collection is not None

is_single_car_dataset = (
    len(files) == 1
    and documents_mod.is_car_schema(pd.read_csv(io.BytesIO(files[0][1])))
    and len(documents_df) == 10000
)
has_car_columns = {"Brand", "Year", "Price", "Fuel_Type", "Transmission"}.issubset(documents_df.columns)

row_counts = documents_df.groupby("source_file").size().to_dict() if "source_file" in documents_df.columns else {}


# --------------------------------------------------------------------------- #
# Filters (shared by Chat + Browse; ignored by Benchmark)
# --------------------------------------------------------------------------- #

def build_filters_from_state(documents_df: pd.DataFrame, state: dict) -> pd.Series:
    """Boolean mask over documents_df. UI state plumbing, not engine logic --
    lives here rather than in rag_engine.py."""
    mask = pd.Series(True, index=documents_df.index)
    if state.get("brands"):
        mask &= documents_df["Brand"].isin(state["brands"])
    if state.get("fuel_types"):
        mask &= documents_df["Fuel_Type"].isin(state["fuel_types"])
    if state.get("transmissions"):
        mask &= documents_df["Transmission"].isin(state["transmissions"])
    if state.get("year_range") and state.get("year_full_range") and state["year_range"] != state["year_full_range"]:
        lo, hi = state["year_range"]
        mask &= documents_df["Year"].between(lo, hi)
    if state.get("price_range") and state.get("price_full_range") and state["price_range"] != state["price_full_range"]:
        lo, hi = state["price_range"]
        mask &= documents_df["Price"].between(lo, hi)
    return mask


filter_state = {}
with st.sidebar:
    st.markdown("""
<div class="sidebar-card">

<div class="sidebar-title">

📁 Data Source

</div>
""", unsafe_allow_html=True)
    for name, content in files:
        n_rows = row_counts.get(name, 0)
        size_kb = len(content) / 1024
        size_str = f"{size_kb/1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
        st.markdown(
            f'<div class="file-chip"><span class="file-chip-icon">📄</span>'
            f'<div><div class="file-chip-name">{html.escape(name)}</div>'
            f'<div class="file-chip-meta">{n_rows:,} rows &middot; {size_str}</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    def side_h(n, label):
        st.markdown(f'<div class="side-h"><span class="num">{n}</span>{label}</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="sidebar-card">
<div class="sidebar-title">
🔍 Retrieval Method
</div>
""", unsafe_allow_html=True)
    retriever_options = ["TF-IDF", "BM25"]
    if embeddings_available:
        retriever_options += ["Embeddings", "Hybrid"]
    if chroma_available:
        retriever_options += ["Chroma (Vector DB)"]
    retriever = st.radio("Method", retriever_options, index=len(retriever_options) - 1, label_visibility="collapsed")

    alpha = 0.6
    if retriever == "Hybrid":
        alpha = st.slider("Semantic weight (α)", 0.0, 1.0, 0.6, 0.05, help="hybrid = α·semantic + (1-α)·lexical")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
<div class="sidebar-card">
<div class="sidebar-title">
⚙️ Search Settings
</div>
""", unsafe_allow_html=True)
    retrieval_k = st.slider("Candidates to retrieve", 3, 15, 8)
    prefer_current = st.toggle("Prefer current over superseded rows", value=True,
                                help="Only meaningful for the car dataset, which has a synthetic freshness flag.")
    max_context_chunks = st.slider("Max sources in context", 1, 6, 3)
    word_budget = st.slider("Word budget (approx.)", 50, 500, 150, 25)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
<div class="sidebar-card">
<div class="sidebar-title">
🎨 Prompt Style
</div>
""", unsafe_allow_html=True)
    prompt_style = st.radio("Style", list(prompting_mod.PROMPT_BUILDERS.keys()), index=2, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
<div class="sidebar-card">
<div class="sidebar-title">
🎯 Filters
</div>
""", unsafe_allow_html=True)
    if has_car_columns:
        brands = sorted(documents_df["Brand"].dropna().unique().tolist())
        fuels = sorted(documents_df["Fuel_Type"].dropna().unique().tolist())
        transmissions = sorted(documents_df["Transmission"].dropna().unique().tolist())
        year_lo, year_hi = int(documents_df["Year"].min()), int(documents_df["Year"].max())
        price_lo, price_hi = int(documents_df["Price"].min()), int(documents_df["Price"].max())

        filter_state["brands"] = st.multiselect("Brand", brands)
        filter_state["fuel_types"] = st.multiselect("Fuel type", fuels)
        filter_state["transmissions"] = st.multiselect("Transmission", transmissions)
        filter_state["year_range"] = st.slider("Year", year_lo, year_hi, (year_lo, year_hi))
        filter_state["year_full_range"] = (year_lo, year_hi)
        filter_state["price_range"] = st.slider("Price ($)", price_lo, price_hi, (price_lo, price_hi), step=100)
        filter_state["price_full_range"] = (price_lo, price_hi)
    else:
        st.caption("Filters are available when the corpus has car-listing columns "
                   "(Brand, Year, Price, Fuel_Type, Transmission).")

filters_active = has_car_columns and any([
    filter_state.get("brands"),
    filter_state.get("fuel_types"),
    filter_state.get("transmissions"),
    filter_state.get("year_range") != filter_state.get("year_full_range"),
    filter_state.get("price_range") != filter_state.get("price_full_range"),
])

if filters_active:
    filter_mask = build_filters_from_state(documents_df, filter_state)
    filtered_document_ids = documents_df.loc[filter_mask, "document_id"].tolist()
    filter_sig = hashlib.sha1(json.dumps({
        k: v for k, v in filter_state.items() if k not in ("year_full_range", "price_full_range")
    }, sort_keys=True, default=str).encode()).hexdigest()
else:
    filtered_document_ids = None
    filter_sig = "ALL"

active_chunks_df, active_vectorizer, active_matrix, active_bm25 = _get_filtered_indexes(
    file_sig, filter_sig, chunks_df, filtered_document_ids
)
active_chunk_embeddings = _embeddings_for_chunks(active_chunks_df, chunk_embeddings) if embeddings_available else None

with st.sidebar:
    if filters_active:
        st.caption(f"Filters active — {len(active_chunks_df):,} / {len(chunks_df):,} rows match.")

st.markdown("</div>", unsafe_allow_html=True)
# --------------------------------------------------------------------------- #
# Retrieval + context + prompt + streaming helpers
# --------------------------------------------------------------------------- #

def retrieve_top_k_chroma(query: str, k: int) -> pd.DataFrame:
    """Query the (full-corpus) Chroma store, then apply the same document-id
    filter the other retrievers get for free from active_chunks_df -- Chroma
    itself is never rebuilt per-filter (see _load_chroma_collection), so
    filtering happens here instead. We over-fetch when filters are active so
    a narrow filter doesn't starve the results down to nothing."""
    pool = k * 5 if filters_active else k
    pool = min(pool, len(chunks_df))
    raw = chroma_mod.query_chroma(chroma_collection, query, top_k=pool, model=embedding_model)

    lookup = chunks_df.set_index("chunk_id")
    ids = [int(i) for i in raw["ids"][0]]
    distances = raw["distances"][0]

    rows = []
    for chunk_id, distance in zip(ids, distances):
        if chunk_id not in lookup.index:
            continue
        row = lookup.loc[chunk_id]
        if filtered_document_ids is not None and row["document_id"] not in filtered_document_ids:
            continue
        row = row.copy()
        row["score"] = 1 - distance  # cosine distance -> similarity
        rows.append(row)
        if len(rows) >= k:
            break

    if not rows:
        results = active_chunks_df.iloc[0:0].copy()
        results["score"] = []
    else:
        results = pd.DataFrame(rows).reset_index()
    results["retriever"] = "Chroma"
    return results.reset_index(drop=True)


def retrieve(query: str) -> pd.DataFrame:
    if retriever == "TF-IDF":
        return retrieve_mod.retrieve_top_k_tfidf(query, active_chunks_df, active_vectorizer, active_matrix, k=retrieval_k)
    if retriever == "BM25":
        return retrieve_mod.retrieve_top_k_bm25(query, active_chunks_df, active_bm25, k=retrieval_k)
    if retriever == "Embeddings":
        return retrieve_mod.retrieve_top_k_semantic(query, active_chunks_df, embedding_model, active_chunk_embeddings, k=retrieval_k)
    if retriever == "Chroma (Vector DB)":
        return retrieve_top_k_chroma(query, retrieval_k)
    return retrieve_mod.retrieve_top_k_hybrid(
        query, active_chunks_df, active_vectorizer, active_matrix, embedding_model, active_chunk_embeddings,
        alpha=alpha, k=retrieval_k,
    )


def render_source_card(row, index) -> str:
    is_car = row.get("schema") == "car" or (row.get("Brand") not in (None, "") and pd.notna(row.get("Brand")))
    if is_car:
        tag_class = "tag-current" if row["is_current"] else "tag-outdated"
        tag_text = "Current" if row["is_current"] else "Outdated"
        title = str(row["title"]).rsplit(" (doc", 1)[0]
        date = row.get("effective_date")
        date_str = str(date.date()) if pd.notna(date) else ""
        return f"""
        <div class="source-card">
            <div class="source-icon-row">
                <span class="source-icon">🚗</span>
                <span class="source-label">Source {index}</span>
            </div>
            <div class="source-title">{html.escape(title)} <span class="source-tag {tag_class}">{tag_text}</span></div>
            <div class="source-price">${row['Price']:,}</div>
            <div class="source-date">{html.escape(date_str)}</div>
        </div>
        """
    return f"""
    <div class="source-card">
        <div class="source-icon-row">
            <span class="source-icon">📄</span>
            <span class="source-label">Source {index}</span>
        </div>
        <div class="source-title">{html.escape(str(row['title']))}</div>
        <div class="source-detail">{html.escape(str(row['chunk_text'])[:140])}{'…' if len(str(row['chunk_text'])) > 140 else ''}</div>
    </div>
    """


def render_source_grid(sources_df: pd.DataFrame, cols_per_row: int = 3):
    rows = list(sources_df.iterrows())
    for start in range(0, len(rows), cols_per_row):
        group = rows[start:start + cols_per_row]
        cols = st.columns(len(group))
        for col, (i, (_, row)) in zip(cols, enumerate(group, start=start + 1)):
            with col:
                st.markdown(render_source_card(row, i), unsafe_allow_html=True)


def render_citations(cited_numbers):
    """Purely self-generated markup (integers we computed, never raw LLM
    text) so this can safely use unsafe_allow_html without widening the
    XSS surface of the answer text itself."""
    if not cited_numbers:
        return
    marks = " ".join(f'<span class="cite-sup">[{n}]</span>' for n in cited_numbers)
    st.markdown(f"Cited: {marks}", unsafe_allow_html=True)


def prepare_answer(query: str):
    """Runs retrieval (or structured aggregation when the query is an
    aggregation AND the corpus has car columns) + context building. Returns
    (package, prompt_text, messages, used_aggregation).

    `messages` is the multi-turn payload for the /chat endpoints -- built
    only when prompt_style is Strict; otherwise it's None and the caller
    falls back to the single-prompt streaming functions.
    """
    agg_kind = aggregation_mod.detect_aggregation(query) if has_car_columns else None
    used_aggregation = False
    if agg_kind is not None:
        agg_results = aggregation_mod.run_aggregation(
            agg_kind, query, documents_df, chunks_df, k=max_context_chunks,
        )
        if len(agg_results) > 0:
            results = agg_results
            used_aggregation = True
        else:
            results = retrieve(query)
    else:
        results = retrieve(query)

    package = retrieve_mod.build_context_package(
        results, max_context_chunks=max_context_chunks, max_chunks_per_document=1,
        word_budget=word_budget, prefer_current=prefer_current,
    )

    # Prepend a [STRUCTURED RESULT] line so the Strict prompt's aggregation
    # rule tells the LLM to treat it as authoritative.
    if used_aggregation:
        summary = aggregation_mod.structured_summary(agg_kind, query, results)
        if summary:
            package["context_text"] = summary + "\n\n" + package["context_text"]

    prompt_text = prompting_mod.PROMPT_BUILDERS[prompt_style](query, package["context_text"])

    messages = None
    if prompt_style == "Strict":
        messages = _build_chat_messages(query, package["context_text"])

    return package, prompt_text, messages, used_aggregation


MAX_HISTORY_TURNS = 6  # cap on prior user/assistant messages sent to the LLM


def _build_chat_messages(current_query: str, current_context: str):
    """Build the messages payload for one Strict-style RAG turn.

    Layout:
      - system : the strict rules (build_strict_system_prompt)
      - last MAX_HISTORY_TURNS entries from st.session_state.messages
        (alternating user/assistant, content-only -- no prior retrieval
        context is replayed; what matters is the latest context)
      - user   : build_user_turn_message(current_query, current_context)
    """
    messages = [{"role": "system", "content": prompting_mod.build_strict_system_prompt()}]

    history = list(st.session_state.get("messages", []))[-MAX_HISTORY_TURNS:]
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({
        "role": "user",
        "content": prompting_mod.build_user_turn_message(current_query, current_context),
    })
    return messages


def stream_or_fallback(package, prompt_text, messages=None):
    """Returns (generator, is_streaming). Retrieval/context already happened
    in prepare_answer -- this only concerns the LLM call itself.

    When `messages` is provided (Strict prompt, multi-turn), the /chat
    endpoints are used so prior turns are visible to the model. Otherwise
    the single-prompt functions are used (Weak / Better / pre-Fix-4 path)."""
    if not package["num_sources"]:
        def _gen():
            yield "I couldn't find anything in the uploaded data relevant to that question."
        return _gen(), False

    if prompting_mod.OPENROUTER_API_KEY:
        def _gen():
            if messages is not None:
                stream = prompting_mod.ask_openrouter_chat_stream(
                    messages,
                    prompting_mod.OPENROUTER_API_KEY,
                    prompting_mod.OPENROUTER_MODEL,
                )
            else:
                stream = prompting_mod.ask_openrouter_stream(
                    prompt_text,
                    prompting_mod.OPENROUTER_API_KEY,
                    prompting_mod.OPENROUTER_MODEL,
                )
            for piece in stream:
                if piece.startswith("__ERROR__"):
                    yield "\n\n_(OpenRouter stream interrupted: " + piece.replace("__ERROR__: ", "") + ")_"
                    return
                yield piece
        return _gen(), True

    if not ollama_ok:
        fallback_text = (
            "No local LLM reachable, so here's the grounded context instead of a generated "
            "answer — open **Prompt used** below and paste it into any chat LLM if you'd like "
            "a written answer.\n\n" + package["context_text"]
        )
        def _gen():
            yield fallback_text
        return _gen(), False

    def _gen():
        if messages is not None:
            stream = prompting_mod.ask_ollama_chat_stream(messages, ollama_host, ollama_model)
        else:
            stream = prompting_mod.ask_ollama_stream(prompt_text, ollama_host, ollama_model)
        for piece in stream:
            if piece.startswith("__ERROR__"):
                yield "\n\n_(stream interrupted: " + piece.replace("__ERROR__: ", "") + ")_"
                return
            yield piece
    return _gen(), True


def render_sources_and_prompt(sources_df, prompt_text, answer_text):
    if len(sources_df):
        if prompt_style == "Strict":
            cited = prompting_mod.parse_cited_sources(answer_text, len(sources_df))
            render_citations(cited)
        st.markdown('<div class="panel-label">Sources</div>', unsafe_allow_html=True)
        render_source_grid(sources_df)
        with st.expander("Prompt used"):
            st.markdown(f'<div class="context-block">{html.escape(prompt_text)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

with tab_chat:
    for msg in st.session_state.messages:
        avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources") is not None and len(msg["sources"]):
                render_sources_and_prompt(msg["sources"], msg["prompt"], msg["content"])

question = st.chat_input("Ask a question about your data...")
st.caption('Tip: Be specific. Example: "Show me all BMW cars with a price under $20,000"')

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with tab_chat:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Retrieving..."):
                package, prompt_text, messages, used_aggregation = prepare_answer(question)
                gen, is_streaming = stream_or_fallback(package, prompt_text, messages)
            if used_aggregation:
                st.markdown(
                    '<span style="background:#EFF4FF;color:#2563EB;padding:2px 8px;'
                    'border-radius:6px;font-size:0.8em;font-weight:600;">'
                    '✓ Structured answer</span>',
                    unsafe_allow_html=True,
                )
            full_text = st.write_stream(gen)
            render_sources_and_prompt(package["selected_df"], prompt_text, full_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "sources": package["selected_df"],
        "prompt": prompt_text,
    })


# --------------------------------------------------------------------------- #
# Browse tab
# --------------------------------------------------------------------------- #

with tab_browse:
    st.markdown("#### Filtered corpus")
    st.caption(f"{len(active_chunks_df):,} of {len(chunks_df):,} rows match the current sidebar filters.")

    display_cols = [c for c in ["title", "Brand", "Model", "Year", "Fuel_Type", "Transmission",
                                 "Mileage", "Owner_Count", "Price", "is_current", "source_file"]
                     if c in active_chunks_df.columns]
    st.dataframe(active_chunks_df[display_cols], width="stretch", height=360)

    if has_car_columns and len(active_chunks_df):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### By brand")
            st.bar_chart(active_chunks_df["Brand"].value_counts())
        with col2:
            st.markdown("##### Price distribution")
            price_bins = pd.cut(active_chunks_df["Price"], bins=10)
            st.bar_chart(price_bins.value_counts().sort_index().rename(lambda x: str(x)))


# --------------------------------------------------------------------------- #
# Benchmark tab (always full corpus -- filters ignored)
# --------------------------------------------------------------------------- #

with tab_benchmark:
    if is_single_car_dataset:
        st.caption(
            "Runs Precision@K / Recall@K / Hit-Rate@K / MRR across the ground-truth query set "
            "used to validate this pipeline against the full car_price_dataset.csv. "
            "**Filters are ignored here** — the ground truth is verified against the complete, unfiltered corpus."
        )
        bench_k = st.slider("K for benchmark", 1, 10, 3, key="bench_k")
        if st.button("Run benchmark"):
            with st.spinner("Evaluating retrievers..."):
                evals = [
                    retrieve_mod.evaluate_retriever("TF-IDF", lambda q, k: retrieve_mod.retrieve_top_k_tfidf(q, chunks_df, tfidf_vectorizer, tfidf_matrix, k), retrieve_mod.GROUND_TRUTH, bench_k),
                    retrieve_mod.evaluate_retriever("BM25", lambda q, k: retrieve_mod.retrieve_top_k_bm25(q, chunks_df, bm25, k), retrieve_mod.GROUND_TRUTH, bench_k),
                ]
                if embeddings_available:
                    evals.append(retrieve_mod.evaluate_retriever("Embeddings", lambda q, k: retrieve_mod.retrieve_top_k_semantic(q, chunks_df, embedding_model, chunk_embeddings, k), retrieve_mod.GROUND_TRUTH, bench_k))
                    evals.append(retrieve_mod.evaluate_retriever("Hybrid", lambda q, k: retrieve_mod.retrieve_top_k_hybrid(q, chunks_df, tfidf_vectorizer, tfidf_matrix, embedding_model, chunk_embeddings, alpha, k), retrieve_mod.GROUND_TRUTH, bench_k))
                if chroma_available:
                    evals.append(
                        retrieve_mod.evaluate_retriever(
                            "Chroma",
                            lambda q, k: retrieve_mod.retrieve_top_k_chroma(
                                q,
                                chunks_df,
                                chroma_collection,
                                embedding_model,
                                k,
                            ),
                            retrieve_mod.GROUND_TRUTH,
                            bench_k,
                        )
                    )
                summary = pd.concat(evals, ignore_index=True).groupby("retriever")[
                    [f"precision@{bench_k}", f"recall@{bench_k}", f"hit_rate@{bench_k}", "reciprocal_rank"]
                ].mean().sort_values("reciprocal_rank", ascending=False)
            st.dataframe(summary, width="stretch")
            st.bar_chart(summary[["reciprocal_rank", f"hit_rate@{bench_k}"]])
    else:
        st.caption("Benchmark is available when the corpus is exactly the bundled car_price_dataset.csv "
                   "(the ground-truth query set is verified against that specific dataset).")

st.caption(
    "Adapted from the Lab 8 / Lab 9 RAG methodology. For the car dataset, `listing_date` and the "
    "current/superseded flag are synthetic (the source CSV has no timestamp) — added to demonstrate "
    "context-building conflict resolution. Aggregation questions (\"cheapest\", \"most expensive\", "
    "etc.) are routed through a structured pandas path (`08_aggregation.py`) that computes the true "
    "min/max over the full corpus; everything else uses similarity retrieval. CSV-derived values in "
    "the UI are HTML-escaped before being interpolated into unsafe_allow_html blocks."
)
