import numpy as np
from sentence_transformers import SentenceTransformer


_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazily loads and returns the embedding model singleton."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embeds a list of texts into a float32 matrix of shape (n, 384).

    Args:
        texts: List of input strings.

    Returns:
        NumPy array of shape (n, 384) with dtype float32.

    Side effects:
        Lazily downloads/loads the embedding model on first call.
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

    Side effects:
        Lazily downloads/loads the embedding model on first call.
    """
    model = _get_model()
    vector = model.encode([query], convert_to_numpy=True)
    return np.asarray(vector, dtype=np.float32)
