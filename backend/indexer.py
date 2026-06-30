"""
indexer.py — FAISS index construction and retrieval for the candidate ranking system.

Why IndexFlatIP?
----------------
We L2-normalise all embeddings before indexing (done in embedder.py).  For unit
vectors, inner product equals cosine similarity, so IndexFlatIP gives exact
cosine nearest-neighbour search with no approximation error — the right trade-off
at 100 k candidates (exact search is fast enough; no need for IVF/HNSW
approximations that would require tuning nprobe / ef_search).

Functions
---------
build_faiss_index  — accept a float32 numpy matrix, return IndexFlatIP
search_index       — query the index and return (distances, indices)
save_index         — write index to disk with faiss.write_index
load_index         — read index from disk with faiss.read_index
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError as _err:
    raise ImportError(
        "faiss-cpu is not installed. Run: pip install faiss-cpu"
    ) from _err

try:
    from config import FAISS_INDEX_PATH, FAISS_TOP_K
except ImportError:
    FAISS_INDEX_PATH = Path("output/candidate_index.faiss")
    FAISS_TOP_K      = 500

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an exact inner-product FAISS index from a pre-normalised embedding
    matrix.

    Parameters
    ----------
    embeddings : float32 ndarray of shape (N, D).
                 Rows MUST be L2-unit-norm (embed_texts() guarantees this).

    Returns
    -------
    faiss.IndexFlatIP with N vectors added, ready for search.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D, got shape {embeddings.shape}"
        )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    n, dim = embeddings.shape

    logger.info("Building IndexFlatIP: %d vectors × %d dims …", n, dim)
    t0 = time.perf_counter()

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info(
        "FAISS index built: %d vectors in %.2f s.  ntotal=%d",
        n, time.perf_counter() - t0, index.ntotal,
    )
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────

def search_index(
    index: faiss.IndexFlatIP,
    jd_embedding: np.ndarray,
    k: int = FAISS_TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the top-k most similar candidates for a given JD embedding.

    Parameters
    ----------
    index        : populated faiss.IndexFlatIP (from build_faiss_index or load_index)
    jd_embedding : float32 ndarray of shape (1, D) or (D,) — unit-norm JD vector
    k            : number of results to retrieve (capped at index.ntotal)

    Returns
    -------
    distances : float32 ndarray of shape (1, k) — cosine similarity scores in [-1, 1]
    indices   : int64  ndarray of shape (1, k) — row indices into the original embedding
                matrix; -1 means "not enough results" (only when k > ntotal)

    Notes
    -----
    • FAISS returns results sorted descending by inner product (highest first).
    • Indices map back to the candidate_ids list held by the embedder:
          top_ids = [candidate_ids[i] for i in indices[0] if i != -1]
    """
    # Ensure (1, D) shape
    vec = np.ascontiguousarray(jd_embedding, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    if vec.ndim != 2 or vec.shape[0] != 1:
        raise ValueError(
            f"jd_embedding must be shape (1, D) or (D,), got {vec.shape}"
        )

    # Re-normalise the query for safety (should already be unit-norm)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    # Cap k at the number of indexed vectors
    k_eff = min(k, index.ntotal)
    if k_eff < k:
        logger.debug("k=%d capped to ntotal=%d", k, index.ntotal)

    logger.debug("Searching FAISS index: k=%d …", k_eff)
    t0 = time.perf_counter()

    distances, indices = index.search(vec, k_eff)

    logger.debug(
        "Search complete in %.3f s.  Top score=%.4f",
        time.perf_counter() - t0,
        float(distances[0, 0]) if k_eff > 0 else float("nan"),
    )
    return distances, indices


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_index(
    index: faiss.IndexFlatIP,
    path: Path | str = FAISS_INDEX_PATH,
) -> None:
    """
    Write the FAISS index to disk using faiss.write_index.

    The parent directory is created if it does not exist.

    Parameters
    ----------
    index : any faiss.Index subclass (IndexFlatIP in our case)
    path  : destination file path (conventionally *.faiss)
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
    logger.info(
        "FAISS index saved → %s  (%d vectors)", out.name, index.ntotal
    )


def load_index(
    path: Path | str = FAISS_INDEX_PATH,
) -> faiss.IndexFlatIP:
    """
    Load a FAISS index from disk.

    Parameters
    ----------
    path : path to the .faiss file written by save_index

    Returns
    -------
    faiss.Index (runtime type matches what was saved, i.e. IndexFlatIP here)

    Raises
    ------
    FileNotFoundError : if path does not exist
    RuntimeError      : if faiss.read_index fails
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"FAISS index not found at: {src}")

    try:
        index = faiss.read_index(str(src))
    except Exception as exc:
        raise RuntimeError(f"Failed to load FAISS index from '{src}': {exc}") from exc

    logger.info(
        "FAISS index loaded ← %s  (%d vectors, dim=%d)",
        src.name, index.ntotal, index.d,
    )
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Utility: convert raw FAISS output to a ranked list of (candidate_id, score)
# ─────────────────────────────────────────────────────────────────────────────

def faiss_results_to_ranked(
    distances: np.ndarray,
    indices: np.ndarray,
    candidate_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Zip FAISS search output with the candidate ID list.

    Parameters
    ----------
    distances     : (1, k) float32 array from search_index
    indices       : (1, k) int64  array from search_index
    candidate_ids : ordered list of IDs as held by embedder

    Returns
    -------
    List of dicts with keys ``candidate_id`` and ``semantic_score``,
    sorted descending by semantic_score.  FAISS sentinel value -1 is dropped.
    """
    results: list[dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "candidate_id"  : candidate_ids[int(idx)],
            "semantic_score": float(dist),   # cosine similarity ∈ [-1, 1]
            "faiss_index"   : int(idx),
        })
    # Already sorted by FAISS (descending), but be explicit
    results.sort(key=lambda r: r["semantic_score"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="Smoke-test FAISS index build + search.")
    ap.add_argument(
        "--index", type=Path, default=FAISS_INDEX_PATH,
        help="Path to a saved FAISS index (skip build if provided and exists)",
    )
    ap.add_argument(
        "--dim", type=int, default=1024,
        help="Embedding dimension for synthetic test (default: 1024 for bge-large)",
    )
    ap.add_argument(
        "--n", type=int, default=1000,
        help="Number of synthetic vectors to index for the test",
    )
    ap.add_argument(
        "--k", type=int, default=10,
        help="Number of results to retrieve in the test search",
    )
    args = ap.parse_args()

    # ── If a real saved index exists, load and search it ─────────────────────
    if Path(args.index).exists():
        idx = load_index(args.index)
        query = np.random.rand(1, idx.d).astype(np.float32)
        query /= np.linalg.norm(query)
        dists, inds = search_index(idx, query, k=args.k)
        print(f"Loaded index with {idx.ntotal:,} vectors (dim={idx.d})")
        print(f"Top-{args.k} distances: {dists[0].tolist()}")
        sys.exit(0)

    # ── Otherwise build a synthetic index as a unit test ─────────────────────
    print(f"Building synthetic index: {args.n} vectors × {args.dim} dims …")
    rng   = np.random.default_rng(42)
    vecs  = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms

    idx   = build_faiss_index(vecs)
    query = vecs[[0]]   # query with the first vector; top hit should be itself

    dists, inds = search_index(idx, query, k=args.k)
    print(f"Top-{args.k} indices : {inds[0].tolist()}")
    print(f"Top-{args.k} scores  : {[round(float(d), 4) for d in dists[0]]}")
    assert inds[0, 0] == 0, "Self-retrieval sanity check failed!"
    print("✓ Self-retrieval check passed (index 0 is its own nearest neighbour).")
