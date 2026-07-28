"""
08_aggregation.py
-----------------
Stage 8 (optional, parallel to retrieval): structured aggregation for queries
that pure similarity retrieval cannot answer correctly.

The similarity retrievers in 06_retrieve_context.py rank chunks by textual
similarity to the query, so a question like "what's the most expensive car in
the lot?" returns chunks whose text happens to look like the words "most",
"expensive", "car" -- not the row with the actual maximum price. The LLM is
then asked to pick the max from a 3-row sample and confidently states a
mid-range number, which is the wrong-answer symptom this module exists to fix.

Approach:
  1. detect_aggregation(query) -> kind | None
     Regex match against the most common aggregation phrasings.
  2. run_aggregation(kind, query, docs_df, k) -> DataFrame
     Filters by brand/model/year mentioned in the query, applies is_current
     when the user says "currently listed", sorts by the relevant numeric
     column, and returns the top-k rows.

The returned DataFrame has the same columns as the chunks the LLM normally
sees (document_id, title, chunk_text, is_current, score, ...) so it slots
into build_context_package unchanged.

Out of scope (intentionally, for the first cut):
  - average/median (LLM can't compute over a sample; structured answer
    would require a synthesized numeric line -- left for a follow-up)
  - count/histogram (same reason)
  - free-form natural-language filter parsing beyond brand/model/year
"""

import re

import pandas as pd

# --------------------------------------------------------------------------- #
# Aggregation intent detection
# --------------------------------------------------------------------------- #

AGG_PATTERNS = {
    "min_price":    r"\b(cheapest|lowest[\s-]?price|least[\s-]?expensive|best[\s-]?price|lowest\s+cost)\b",
    "max_price":    r"\b(most\s+expensive|priciest|highest[\s-]?price|dearest|top[\s-]?dollar)\b",
    "min_mileage":  r"\b(fewest\s+miles|lowest\s+mileage|lowest\s+odometer|least\s+driven|lowest\s+kms?)\b",
    "max_mileage":  r"\b(most\s+miles|highest\s+mileage|highest\s+odometer|most\s+driven|highest\s+kms?)\b",
    "oldest":       r"\b(oldest|earliest\s+year|first[\s-]?made)\b",
    "newest":       r"\b(newest|latest\s+year|most\s+recent|brand[\s-]?new)\b",
}

# Maps an aggregation kind to the (column, ascending) used for sorting.
AGG_SORT = {
    "min_price":    ("Price", True),
    "max_price":    ("Price", False),
    "min_mileage":  ("Mileage", True),
    "max_mileage":  ("Mileage", False),
    "oldest":       ("Year", True),
    "newest":       ("Year", False),
}

CURRENT_LISTING_RE = re.compile(
    r"\b(currently\s+(listed|in\s+(the\s+)?(lot|inventory|stock)|available|for\s+sale)|"
    r"current\s+(inventory|stock|lot|listings?|asking\s+price)|"
    r"on\s+the\s+lot\s+right\s+now|right\s+now|available\s+now)\b",
    re.IGNORECASE,
)


def detect_aggregation(query: str):
    """Returns the aggregation kind key (e.g. 'min_price'), or None if the
    query does not look like a supported aggregation."""
    if not query:
        return None
    for kind, pattern in AGG_PATTERNS.items():
        if re.search(pattern, query, flags=re.IGNORECASE):
            return kind
    return None


# --------------------------------------------------------------------------- #
# Filter extraction (brand / model / year)
# --------------------------------------------------------------------------- #

def _extract_brand(query: str, available_brands):
    """Match a known brand mentioned in the query. Case-insensitive,
    word-boundary aware so 'Audi' doesn't match 'Auditorium'."""
    if not available_brands:
        return None
    brands_sorted = sorted(available_brands, key=len, reverse=True)
    for brand in brands_sorted:
        if not isinstance(brand, str) or not brand.strip():
            continue
        if re.search(rf"\b{re.escape(brand)}\b", query, flags=re.IGNORECASE):
            return brand
    return None


def _extract_year_range(query: str):
    """Pull things like 'from 2020', 'since 2018', '2020-2023', 'after 2019'."""
    years = []
    # Range: 2020-2023 or 2020 to 2023
    range_match = re.search(r"\b(19|20)(\d{2})\s*[-–to]+\s*(19|20)?(\d{2})\b", query)
    if range_match:
        try:
            start = int((range_match.group(1) or "") + range_match.group(2))
            end_prefix = range_match.group(3) or range_match.group(1)
            end = int(end_prefix + range_match.group(4))
            years.extend([start, end])
        except ValueError:
            pass
    if years:
        return min(years), max(years)

    # Single year with preposition: from/since/after/before 2018
    m = re.search(r"\b(?:from|since|after)\s+(19|20)(\d{2})\b", query, flags=re.IGNORECASE)
    if m:
        y = int(m.group(1) + m.group(2))
        return y, None
    m = re.search(r"\b(?:before|until|up\s+to)\s+(19|20)(\d{2})\b", query, flags=re.IGNORECASE)
    if m:
        y = int(m.group(1) + m.group(2))
        return None, y
    return None, None


# --------------------------------------------------------------------------- #
# Run aggregation
# --------------------------------------------------------------------------- #

def run_aggregation(kind: str, query: str, docs_df: pd.DataFrame, chunks_df: pd.DataFrame, k: int = 3):
    """Compute the actual top-k rows for an aggregation.

    Parameters
    ----------
    kind : str
        Key from AGG_PATTERNS (e.g. 'min_price').
    query : str
        Original user query (used to extract brand/year filters).
    docs_df : DataFrame
        documents_df from 01_documents -- has Brand/Year/Price/Mileage/is_current.
    chunks_df : DataFrame
        chunks from 03_chunking -- used to pull chunk_text for each top doc.
    k : int
        Number of top rows to return.

    Returns
    -------
    DataFrame with the columns build_context_package expects (document_id,
    title, chunk_text, is_current, score) so it slots into the existing flow
    unchanged. Returns an empty DataFrame if no rows match.
    """
    if kind not in AGG_SORT:
        return chunks_df.iloc[0:0].copy()
    if docs_df is None or len(docs_df) == 0:
        return chunks_df.iloc[0:0].copy()

    sort_col, ascending = AGG_SORT[kind]
    if sort_col not in docs_df.columns:
        return chunks_df.iloc[0:0].copy()

    df = docs_df.copy()

    # Apply is_current filter when the user is asking about "current inventory".
    if "is_current" in df.columns and CURRENT_LISTING_RE.search(query):
        df = df[df["is_current"]]

    # Brand filter.
    if "Brand" in df.columns:
        brand = _extract_brand(query, df["Brand"].dropna().unique().tolist())
        if brand is not None:
            df = df[df["Brand"].astype(str).str.lower() == brand.lower()]

    # Year filter (only meaningful for price/mileage aggregations, not for
    # oldest/newest where Year IS the sort key and filtering it would change
    # the question).
    if "Year" in df.columns and kind not in ("oldest", "newest"):
        year_lo, year_hi = _extract_year_range(query)
        if year_lo is not None:
            df = df[df["Year"] >= year_lo]
        if year_hi is not None:
            df = df[df["Year"] <= year_hi]

    if len(df) == 0:
        return chunks_df.iloc[0:0].copy()

    df = df.sort_values(by=sort_col, ascending=ascending).head(k)

    # Join chunk_text back from chunks_df. Each document has at least one chunk;
    # pick the first chunk per document to represent it.
    first_chunk = (
        chunks_df.sort_values(by=["document_id", "chunk_id"])
        .drop_duplicates(subset="document_id", keep="first")
    )
    result = df.merge(first_chunk[["document_id", "chunk_text", "chunk_id", "title"]], on="document_id", how="left")
    # chunks_df title doesn't always survive the merge cleanly (documents_df
    # already has its own title); prefer the documents_df title when present.
    if "title_x" in result.columns:
        result = result.rename(columns={"title_x": "title"}).drop(columns=["title_y"], errors="ignore")

    # score: descending 1.0, 0.9, ... so build_context_package's score sort
    # preserves the aggregation's ranking (it sorts by [is_current, score]
    # and we want the agg order to win within each current-ness bucket).
    result["score"] = [1.0 - 0.01 * i for i in range(len(result))]
    result["retriever"] = f"Aggregation/{kind}"

    return result.reset_index(drop=True)


def structured_summary(kind: str, query: str, top_rows: pd.DataFrame):
    """A short, LLM-readable summary line of what the structured path
    computed. Prepended to the context block so the LLM can cite it."""
    if top_rows is None or len(top_rows) == 0:
        return ""
    label = {
        "min_price":   "Lowest price",
        "max_price":   "Highest price",
        "min_mileage": "Lowest mileage",
        "max_mileage": "Highest mileage",
        "oldest":      "Oldest",
        "newest":      "Newest",
    }.get(kind, "Top result")
    rows_desc = "; ".join(
        f"doc {int(r['document_id'])} — {r.get('title', '').rsplit(' (doc', 1)[0]} "
        f"(${int(r['Price']):,})"
        for _, r in top_rows.iterrows()
        if pd.notna(r.get("Price"))
    )
    return f"[STRUCTURED RESULT] {label} (matches your query '{query.strip()}'): {rows_desc}"


if __name__ == "__main__":
    # Lightweight self-test for the regex patterns.
    test_queries = [
        ("What's the cheapest Audi currently in the lot?", "min_price"),
        ("Which car is the most expensive in the current inventory?", "max_price"),
        ("Which currently listed car has the lowest odometer reading?", "min_mileage"),
        ("Which currently listed car has been driven the most miles?", "max_mileage"),
        ("What's the oldest BMW still on the lot?", "oldest"),
        ("Show me the newest electric car.", "newest"),
        ("Is there a one-owner electric BMW X5 from 2018 or later?", None),
    ]
    print("Aggregation detector self-test:")
    for q, expected in test_queries:
        got = detect_aggregation(q)
        status = "OK" if got == expected else "FAIL"
        print(f"  [{status}] expected={expected!r:>12} got={got!r:>12}  | {q}")
