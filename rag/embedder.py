import os
import numpy as np
from typing import Any

_model: Any = None

def _get_model() -> Any:
    """Lazily loads and returns the embedding model singleton with strict memory limits."""
    global _model
    if _model is None:
        print("Loading local PyTorch model...")
        # Restrict threads to prevent memory spikes on Render free tier
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        import torch
        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embeds a list of texts into a float32 matrix of shape (n, 384).

    Args:
        texts: List of input strings.

    Returns:
        NumPy array of shape (n, 384) with dtype float32.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32)

def embed_query(query: str) -> np.ndarray:
    """
    Embeds one query into a float32 matrix of shape (1, 384).

    Args:
        query: User query string.

    Returns:
        NumPy array of shape (1, 384) with dtype float32.
    """
    model = _get_model()
    vector = model.encode([query], convert_to_numpy=True)
    return np.asarray(vector, dtype=np.float32)
