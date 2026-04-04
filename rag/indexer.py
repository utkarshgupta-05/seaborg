import os
import sys

import faiss
import pandas as pd
from dotenv import load_dotenv

from .embedder import embed_texts
from .summariser import summarise_row


def build_and_save() -> None:
    """
    Builds a FAISS IndexFlatL2(384) from all rows in parquet and saves it to disk.

    Args:
        None.

    Returns:
        None.

    Side effects:
        Reads parquet data, computes embeddings, and writes a FAISS index file.
    """
    load_dotenv()
    parquet_path = os.getenv("PARQUET_PATH")
    faiss_index_path = os.getenv("FAISS_INDEX_PATH")

    if not parquet_path:
        raise ValueError("PARQUET_PATH is not set.")
    if not faiss_index_path:
        raise ValueError("FAISS_INDEX_PATH is not set.")

    df = pd.read_parquet(parquet_path)
    summaries = [summarise_row(row) for _, row in df.iterrows()]
    vectors = embed_texts(summaries)
    vectors = vectors.astype("float32")

    index = faiss.IndexFlatL2(384)
    if vectors.shape[1] != 384:
        raise ValueError(f"Embedding dimension mismatch: expected 384, got {vectors.shape[1]}")
    index.add(vectors)

    faiss.write_index(index, faiss_index_path)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"Indexed {len(summaries)} vectors → saved to {faiss_index_path}")
