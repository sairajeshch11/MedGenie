from __future__ import annotations

import os, json, pickle
from typing import Any, Dict, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

FAISS_PATH = os.path.join(INDEX_DIR, "faiss_final.index")
BM25_PATH  = os.path.join(INDEX_DIR, "bm25_final.pkl")
CHUNKS_PKL = os.path.join(INDEX_DIR, "chunks.pkl")
CORPUS_TXT = os.path.join(INDEX_DIR, "corpus_path.txt")

# IMPORTANT: Must match the model used to BUILD the FAISS index
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim

_model = None
_index = None
_bm25 = None
_chunks = None

def _read_corpus_path() -> str:
    if os.path.exists(CORPUS_TXT):
        with open(CORPUS_TXT, "r", encoding="utf-8") as f:
            p = f.read().strip()
        if p:
            return p
    # fallback to common location
    fallback = os.path.join(BASE_DIR, "data", "medical_chunks_final.jsonl")
    return fallback

def load_corpus() -> List[Dict[str, Any]]:
    corpus_path = _read_corpus_path()
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus jsonl not found: {corpus_path}")
    items: List[Dict[str, Any]] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t.strip()]

def load_indexes() -> None:
    global _model, _index, _bm25, _chunks

    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)

    if _index is None:
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(f"FAISS index not found: {FAISS_PATH}")
        _index = faiss.read_index(FAISS_PATH)

        # hard guard: ensure embedding dim matches FAISS dim
        test = _model.encode(["dim check"], normalize_embeddings=True)
        qdim = int(np.asarray(test).shape[1])
        if qdim != _index.d:
            raise ValueError(f"Embedding dim mismatch: model={qdim} vs faiss={_index.d}. "
                             f"Fix EMBED_MODEL_NAME or rebuild index.")

    if _bm25 is None or _chunks is None:
        if os.path.exists(BM25_PATH) and os.path.exists(CHUNKS_PKL):
            with open(BM25_PATH, "rb") as f:
                _bm25 = pickle.load(f)
            with open(CHUNKS_PKL, "rb") as f:
                _chunks = pickle.load(f)
        else:
            # fallback: build BM25 from corpus if pickles missing
            _chunks = load_corpus()
            corpus_tokens = [_tokenize(c.get("text","")) for c in _chunks]
            _bm25 = BM25Okapi(corpus_tokens)

def retrieve_with_meta(query: str, top_k: int = 3, cand_k: int = 12) -> Dict[str, Any]:
    load_indexes()
    assert _model is not None and _index is not None and _bm25 is not None and _chunks is not None

    # Dense retrieval
    q = _model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    D, I = _index.search(q, cand_k)

    dense_ids = I[0].tolist()
    dense_scores = D[0].tolist()

    # BM25 retrieval across full corpus (cheap enough for ~1.5k chunks)
    tokenized_query = _tokenize(query)
    bm25_scores = _bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::-1][:cand_k].tolist()

    # Normalize scores for mixing
    def _norm(arr):
        arr = np.asarray(arr, dtype="float32")
        if arr.size == 0:
            return arr
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    dense_norm = _norm(dense_scores)
    bm25_norm_full = _norm(bm25_scores)

    # Hybrid combine on union set
    union = {}
    for rank, idx in enumerate(dense_ids):
        union[idx] = union.get(idx, 0.0) + float(dense_norm[rank]) * 0.65
    for idx in bm25_top:
        union[idx] = union.get(idx, 0.0) + float(bm25_norm_full[idx]) * 0.35

    # Sort by combined score
    ranked = sorted(union.items(), key=lambda x: x[1], reverse=True)[:top_k]

    chunks = []
    for idx, _score in ranked:
        if 0 <= idx < len(_chunks):
            chunks.append(_chunks[idx])

    # Diagnostics
    agreement = len(set(dense_ids[:top_k]).intersection(set(bm25_top[:top_k]))) / max(1, top_k)
    diag = {
        "dense_strength": float(np.mean(dense_norm[:top_k])) if len(dense_norm) else 0.0,
        "bm25_strength": float(np.mean([bm25_scores[i] for i in bm25_top[:top_k]])) if bm25_top else 0.0,
        "agreement": float(agreement),
    }

    return {"chunks": chunks, "diag": diag}
