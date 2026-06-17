import os
import json
import urllib.request
import urllib.error
import socket
import numpy as np
import time
from typing import Any

_model: Any = None

def _get_model() -> Any:
    """Lazily loads and returns the embedding model singleton with strict memory limits."""
    global _model
    if _model is None:
        print("Loading local PyTorch model (API failed or not configured)...")
        # Restrict threads to prevent memory spikes on Render free tier
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        import torch
        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model

def _embed_via_api(texts: list[str]) -> np.ndarray:
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN is not set.")
        
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = json.dumps({"inputs": texts}).encode("utf-8")
    
    _orig_getaddrinfo = socket.getaddrinfo
    
    def _doh_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if host == "api-inference.huggingface.co":
            try:
                # Use Google DoH to bypass Render DNS failures
                req = urllib.request.Request(f"https://dns.google/resolve?name={host}&type=A")
                with urllib.request.urlopen(req, timeout=5.0) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    for answer in data.get("Answer", []):
                        if answer.get("type") == 1:
                            ip = answer.get("data")
                            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, port))]
            except Exception as e:
                print(f"DoH resolution failed: {e}")
        return _orig_getaddrinfo(host, port, family, type, proto, flags)
    
    socket.getaddrinfo = _doh_getaddrinfo

    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30.0) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    if isinstance(result, dict) and "error" in result:
                        # If model is loading, wait and retry
                        if "loading" in result["error"].lower() and attempt < max_retries - 1:
                            time.sleep(15)
                            continue
                        raise RuntimeError(f"API Error: {result['error']}")
                    return np.asarray(result, dtype=np.float32)
            except urllib.error.HTTPError as e:
                error_msg = e.read().decode("utf-8")
                if e.code == 503 and attempt < max_retries - 1:
                    # 503 Service Unavailable is common when HF model is loading
                    time.sleep(15)
                    continue
                raise RuntimeError(f"HF API failed: {e.code} - {error_msg}")
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                raise RuntimeError(f"HF API request failed: {e}")
    finally:
        socket.getaddrinfo = _orig_getaddrinfo

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

    if os.getenv("HUGGINGFACE_TOKEN"):
        # We explicitly don't catch and fallback here because local PyTorch 
        # causes Memory Spikes (OOM) on the Render free tier.
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
    if os.getenv("HUGGINGFACE_TOKEN"):
        # We explicitly don't catch and fallback here because local PyTorch 
        # causes Memory Spikes (OOM) on the Render free tier.
        return _embed_via_api([query])

    model = _get_model()
    vector = model.encode([query], convert_to_numpy=True)
    return np.asarray(vector, dtype=np.float32)
