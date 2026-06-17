import numpy as np
import os
import httpx
from typing import Any

_model: Any = None


def _get_model() -> Any:
    """Lazily loads and returns the embedding model singleton."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _embed_via_api(texts: list[str]) -> np.ndarray:
    """Uses Hugging Face Inference API to generate embeddings to save memory."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN is not set.")
        
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = httpx.post(api_url, headers=headers, json={"inputs": texts}, timeout=30.0)
    response.raise_for_status()
    
    return np.asarray(response.json(), dtype=np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embeds a list of texts into a float32 matrix of shape (n, 384).

    Args:
        texts: List of input strings.

    Returns:
        NumPy array of shape (n, 384) with dtype float32.

    Side effects:
        Calls HF API if in production, else lazily loads local model.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    # Use cloud API in production to prevent Render Out-Of-Memory crashes
    if os.getenv("HUGGINGFACE_TOKEN") and os.getenv("ENVIRONMENT") == "production":
        return _embed_via_api(texts)

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

    Side effects:
        Calls HF API if in production, else lazily loads local model.
    """
    # Use cloud API in production to prevent Render Out-Of-Memory crashes
    if os.getenv("HUGGINGFACE_TOKEN") and os.getenv("ENVIRONMENT") == "production":
        return _embed_via_api([query])

    model = _get_model()
    vector = model.encode([query], convert_to_numpy=True)
    return np.asarray(vector, dtype=np.float32)

