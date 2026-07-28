"""
01_documents.py
----------------
Stage 1 of the RAG pipeline: raw CSV rows -> RAG "documents".

Moved from rag_engine.py, unchanged, per the refactor mapping:
    build_documents, CAR_SCHEMA_COLUMNS, is_car_schema,
    build_documents_generic, build_documents_any, combine_documents
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1. CSV -> Documents
# --------------------------------------------------------------------------- #

def build_documents(raw_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Turn raw car-listing rows into RAG documents with synthetic listing_date
    and a current/superseded flag (the CSV has no date column, so this is a
    clearly-synthetic addition made to demonstrate current-vs-outdated context
    building, exactly like the policy documents in Lab 8/9)."""
    rng = np.random.default_rng(seed)

    df = raw_df.reset_index(drop=True).copy()
    df["document_id"] = df.index

    base_date = pd.Timestamp("2026-07-01")
    days_back = rng.integers(0, 730, size=len(df))
    df["listing_date"] = base_date - pd.to_timedelta(days_back, unit="D")

    dup_key = ["Brand", "Model", "Year", "Fuel_Type", "Transmission"]
    df["is_current"] = True
    rank_in_group = df.groupby(dup_key)["listing_date"].rank(method="first", ascending=False)
    df.loc[rank_in_group > 1, "is_current"] = False

    def row_to_text(row):
        status = (
            "Active listing."
            if row["is_current"]
            else "Superseded listing (a newer listing exists for this same "
                 "configuration; price and mileage below may be out of date)."
        )
        return (
            f"{row['Year']} {row['Brand']} {row['Model']}, {row['Doors']}-door "
            f"{row['Fuel_Type'].lower()} vehicle with a {row['Engine_Size']}L engine "
            f"and {row['Transmission'].lower()} transmission. "
            f"Odometer reading: {row['Mileage']:,} miles. "
            f"Ownership history: {row['Owner_Count']} previous owner(s). "
            f"Listed price: ${row['Price']:,}. "
            f"Listing status: {status} Listing date: {row['listing_date'].date()}."
        )

    df["text"] = df.apply(row_to_text, axis=1)
    df["title"] = (
        df["Year"].astype(str) + " " + df["Brand"] + " " + df["Model"]
        + " (doc " + df["document_id"].astype(str) + ")"
    )
    df["doc_type"] = "car_listing"
    df["search_text"] = df["title"] + " " + df["Brand"] + " " + df["Model"] + " " + df["text"]

    return df[[
        "document_id", "title", "doc_type", "Brand", "Model", "Year", "Engine_Size",
        "Fuel_Type", "Transmission", "Mileage", "Doors", "Owner_Count", "Price",
        "listing_date", "is_current", "text", "search_text",
    ]]


CAR_SCHEMA_COLUMNS = {
    "Brand", "Model", "Year", "Engine_Size", "Fuel_Type",
    "Transmission", "Mileage", "Doors", "Owner_Count", "Price",
}

# Common header variants seen in the wild. The bundled car_price_dataset.csv
# uses `rand` (a typo of `Brand`) as its manufacturer-column header; without
# this alias, is_car_schema() returns False and the whole car-specific path
# is skipped, losing structured fields, listing_date, and the current/
# superseded flag.
COLUMN_ALIASES = {
    "rand": "Brand",
    "Make": "Brand",
    "Manufacturer": "Brand",
}


def is_car_schema(df: pd.DataFrame) -> bool:
    return CAR_SCHEMA_COLUMNS.issubset(set(df.columns))


def build_documents_generic(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Row -> document for an arbitrary CSV whose schema we don't know ahead of
    time. Each row becomes a 'field: value' sentence so it reads naturally for
    a retriever and an LLM, instead of a synthesized car-specific sentence."""
    df = df.reset_index(drop=True).copy()
    df["document_id"] = df.index
    df["source_file"] = source_name

    text_cols = [c for c in df.columns if c not in ("document_id", "source_file")]

    def row_to_text(row):
        parts = [f"{col}: {row[col]}" for col in text_cols if pd.notna(row[col]) and str(row[col]).strip() != ""]
        return "; ".join(parts) + "."

    df["text"] = df.apply(row_to_text, axis=1)
    # Use the first couple of columns as a human-readable title fallback.
    title_cols = text_cols[:2]
    df["title"] = df.apply(lambda r: " ".join(str(r[c]) for c in title_cols) + f" (doc {r['document_id']})", axis=1)
    df["doc_type"] = "csv_row"
    df["search_text"] = df["title"] + " " + df["text"]
    df["is_current"] = True
    df["listing_date"] = pd.NaT

    keep = ["document_id", "source_file", "title", "doc_type", "text", "search_text", "is_current", "listing_date"]
    return df[keep]


def build_documents_any(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Dispatch to the car-specific builder when the schema matches, otherwise
    fall back to the generic builder. Always returns a uniform set of columns
    (schema-specific extras are added as NaN so downstream code never has to
    special-case missing columns) plus a 'schema' tag the UI uses for rendering.

    Column aliases are applied first so a CSV whose brand column is named
    `rand` (as in the bundled car_price_dataset.csv) is still detected as a
    car listing and routed through build_documents instead of the generic
    fallback."""
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    if is_car_schema(df):
        docs = build_documents(df)
        docs["source_file"] = source_name
        docs["schema"] = "car"
        return docs
    docs = build_documents_generic(df, source_name)
    docs["schema"] = "generic"
    return docs


def combine_documents(doc_frames):
    """Concatenate documents built from multiple uploaded CSVs into one corpus
    with document_id re-indexed to stay unique across files."""
    combined = pd.concat(doc_frames, ignore_index=True, sort=False)
    combined["document_id"] = combined.index
    return combined
