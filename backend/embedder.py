"""
embedder.py — Offline BGE embedding layer for the candidate ranking system.

Responsibilities
----------------
• Load SentenceTransformer once and keep it in module-level state (no reload).
• embed_texts()  — batch-encode a list of strings → L2-normalised float32 matrix.
• embed_jd()     — build a single rich string from a JDProfile and embed it.
• Disk cache     — embeddings_cache.npy + candidate_ids_cache.npy written to
                   OUTPUT_DIR so re-runs skip the ~10-min encode step.

__main__ block
--------------
  python embedder.py [--jsonl path] [--force]

  Streams candidates → extracts embedding texts → encodes in batches →
  writes cache + FAISS index → prints "Indexed N candidates".
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

# Let torch use all available CPU cores for matmuls during encoding.
# sentence-transformers/torch defaults can under-use available cores in some
# environments; setting this explicitly before the model loads gives a
# meaningful speedup on multi-core machines with no downside on single-core
# ones (torch clamps to what's actually available).
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

try:
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # handled by the offline fallback in _get_model()

try:
    from config import (
        BATCH_SIZE,
        CANDIDATES_PATH,
        EMBEDDING_MODEL,
        EMBEDDINGS_PATH,
        FAISS_INDEX_PATH,
        OUTPUT_DIR,
    )
except ImportError:
    EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
    BATCH_SIZE       = 256
    OUTPUT_DIR       = Path("output")
    EMBEDDINGS_PATH  = OUTPUT_DIR / "candidate_embeddings.npy"
    FAISS_INDEX_PATH = OUTPUT_DIR / "candidate_index.faiss"
    CANDIDATES_PATH  = Path("data/candidates.jsonl")

logger = logging.getLogger(__name__)

# ── Derived cache paths (sit next to embeddings) ──────────────────────────────
_IDS_CACHE_PATH  = EMBEDDINGS_PATH.parent / "candidate_ids_cache.npy"
_EMBS_CACHE_PATH = EMBEDDINGS_PATH          # alias for clarity below
_FALLBACK_ENCODER_CACHE_PATH = EMBEDDINGS_PATH.parent / "tfidf_fallback_encoder.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Offline fallback encoder
# ─────────────────────────────────────────────────────────────────────────────
# submission_spec.md Section 3 requires the RANKING step to run with network
# OFF, CPU-only, inside a sandboxed container. `SentenceTransformer(model_name)`
# downloads model weights from HuggingFace on first use — fine if the weights
# are pre-cached into the image/container before ranking runs (a documented
# pre-computation step per Section 10.3), but if the Stage-3 reproduction
# sandbox truly has no network at all and no pre-baked model cache, that first
# download will hang or fail, and the whole pipeline goes down with it.
#
# This fallback makes the pipeline degrade gracefully instead of crashing:
# if BGE can't be loaded for any reason (package missing, no network, HF
# unreachable), we transparently switch to a pure scikit-learn TF-IDF +
# Truncated-SVD dense encoder. It's lower quality than BGE (no semantic
# generalisation beyond shared vocabulary) but it's 100% local, has zero
# network dependency, and keeps semantic retrieval functional end-to-end.
#
# Recommended: pre-download BGE and commit/cache the weights (or the
# candidate_embeddings.npy artifact itself) so the real run always uses BGE;
# treat this fallback as a safety net, not the primary path.

_USING_FALLBACK_EMBEDDER = False


class _TfidfFallbackEncoder:
    """Drop-in stand-in for SentenceTransformer.encode(), offline-only."""

    def __init__(self, n_components: int = 384) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(
            max_features=50_000, ngram_range=(1, 2), stop_words="english",
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        self._fitted = False

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,   # unused — kept for interface parity
        show_progress_bar: bool = False,  # unused
        convert_to_numpy: bool = True,    # unused — always returns numpy
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if not self._fitted:
            # IMPORTANT: this must be fit on the full candidate corpus, not
            # on whatever `texts` happens to be passed first. If the
            # candidate-embeddings cache is hit (common — that's the whole
            # point of the cache), this encoder is never fit on the corpus
            # in the current process at all; if the *next* caller happens to
            # be embed_jd() with a single JD string, fit_transform on 1
            # document produces a near-empty vocabulary and SVD blows up
            # ("n_components must be <= n_features"). See save()/load().
            tfidf = self.vectorizer.fit_transform(texts)
            dense = self.svd.fit_transform(tfidf)
            self._fitted = True
        else:
            tfidf = self.vectorizer.transform(texts)
            dense = self.svd.transform(tfidf)

        dense = np.asarray(dense, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            dense = dense / norms
        return dense

    def save(self, path: Path) -> None:
        import pickle
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "_TfidfFallbackEncoder | None":
        import pickle
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, _TfidfFallbackEncoder) and obj._fitted:
                return obj
        except Exception as exc:
            logger.warning("Could not load cached fallback encoder (%s); refitting.", exc)
        return None


try:
    from config import RANDOM_SEED
except ImportError:
    RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Model singleton
# ─────────────────────────────────────────────────────────────────────────────

_model: Any = None


def _get_model(model_name: str = EMBEDDING_MODEL) -> Any:
    """
    Return (and lazily initialise) the module-level embedding model.
    Loading only happens on the first call; subsequent calls return the
    already-loaded instance. Falls back to an offline TF-IDF+SVD encoder
    (see `_TfidfFallbackEncoder`) if BGE can't be loaded for any reason.
    """
    global _model, _USING_FALLBACK_EMBEDDER
    if _model is not None:
        return _model

    if SentenceTransformer is not None:
        try:
            logger.info("Loading embedding model '%s' …", model_name)
            t0 = time.perf_counter()
            _model = SentenceTransformer(model_name)
            logger.info("Model loaded in %.1f s.", time.perf_counter() - t0)
            return _model
        except Exception as exc:
            logger.warning(
                "Could not load '%s' (%s). This usually means no network "
                "access to HuggingFace. Falling back to an offline TF-IDF+SVD "
                "embedder so the pipeline still runs end-to-end — quality "
                "will be lower than BGE. Pre-download/cache the BGE weights "
                "before the ranking step to avoid this in production.",
                model_name, exc,
            )
    else:
        logger.warning(
            "sentence-transformers is not installed. Falling back to an "
            "offline TF-IDF+SVD embedder so the pipeline still runs "
            "end-to-end — quality will be lower than BGE. "
            "Run: pip install sentence-transformers"
        )

    _USING_FALLBACK_EMBEDDER = True
    _model = _TfidfFallbackEncoder.load(_FALLBACK_ENCODER_CACHE_PATH) or _TfidfFallbackEncoder()
    return _model


def using_fallback_embedder() -> bool:
    """True if the module fell back to the offline TF-IDF encoder."""
    return _USING_FALLBACK_EMBEDDER


# ─────────────────────────────────────────────────────────────────────────────
# Core embedding functions
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    model_name: str = EMBEDDING_MODEL,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode a list of strings and return a float32 matrix of shape
    (len(texts), embedding_dim) with each row L2-normalised to unit length.

    L2 normalisation is essential: we use FAISS IndexFlatIP (inner product)
    as a proxy for cosine similarity, which only holds when vectors are
    unit-norm.

    Parameters
    ----------
    texts         : list of strings to encode (empty strings are handled)
    batch_size    : sentences per inference batch (tune to CPU RAM)
    model_name    : override the default EMBEDDING_MODEL
    show_progress : show tqdm progress bar during encoding

    Returns
    -------
    np.ndarray of shape (N, D), dtype float32, each row unit-norm.
    """
    if not texts:
        raise ValueError("embed_texts received an empty list.")

    # Replace empty strings to avoid tokeniser warnings
    clean = [t if t.strip() else "[EMPTY]" for t in texts]

    model = _get_model(model_name)

    logger.info("Encoding %d texts in batches of %d …", len(clean), batch_size)
    t0 = time.perf_counter()

    # SentenceTransformer.encode returns float32 numpy by default
    embeddings: np.ndarray = model.encode(
        clean,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise in-model → cosine via IP
    )

    logger.info(
        "Encoded %d texts → shape %s in %.1f s.",
        len(clean), embeddings.shape, time.perf_counter() - t0,
    )

    # Defensive re-normalise (in case the model flag is ignored by older versions)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid divide-by-zero on zero vecs
    embeddings = (embeddings / norms).astype(np.float32)

    return embeddings


def embed_jd(jd_profile: dict[str, Any], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Build a single descriptive string from a JDProfile and embed it.

    The string template mirrors the `build_text_for_embedding` format used for
    candidates so that cosine similarity is computed in a comparable space.

    Returns
    -------
    np.ndarray of shape (1, D), dtype float32, unit-norm.
    """
    parts: list[str] = []

    # Role title + seniority
    title    = jd_profile.get("role_title", "")
    seniority = jd_profile.get("seniority", "")
    if title:
        parts.append(f"{seniority} {title}".strip() if seniority else title)

    # Experience expectation
    exp = jd_profile.get("experience")
    if exp and isinstance(exp, dict):
        raw = exp.get("raw", "")
        if raw:
            parts.append(f"Experience: {raw}")

    # Mandatory skills
    mandatory = jd_profile.get("mandatory_skills", []) or []
    if mandatory:
        parts.append("Required skills: " + ", ".join(mandatory))

    # Preferred skills
    preferred = jd_profile.get("preferred_skills", []) or []
    if preferred:
        parts.append("Preferred skills: " + ", ".join(preferred))

    # Locations
    locations = jd_profile.get("locations", []) or []
    if locations:
        parts.append("Location: " + ", ".join(locations))

    # Fallback: use raw text (first 600 chars) if all structured fields empty
    if not parts:
        raw_text = (jd_profile.get("raw_text") or "")[:600]
        if raw_text:
            parts.append(raw_text)

    jd_text = " | ".join(parts)
    logger.debug("JD embedding string (%d chars): %s …", len(jd_text), jd_text[:120])

    vec = embed_texts([jd_text], model_name=model_name)   # shape (1, D)
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Disk cache
# ─────────────────────────────────────────────────────────────────────────────

def save_embeddings_cache(
    embeddings: np.ndarray,
    candidate_ids: list[str],
    embs_path: Path = _EMBS_CACHE_PATH,
    ids_path: Path  = _IDS_CACHE_PATH,
) -> None:
    """
    Persist embeddings matrix and corresponding candidate ID array to disk.

    Files
    -----
    embs_path  : float32 numpy array  (N, D)
    ids_path   : object/str numpy array (N,)  — preserves insertion order
    """
    embs_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(embs_path), embeddings.astype(np.float32))
    np.save(str(ids_path),  np.array(candidate_ids, dtype=object))
    logger.info(
        "Cache saved → %s  (%s) and %s  (%d ids)",
        embs_path.name, embeddings.shape,
        ids_path.name,  len(candidate_ids),
    )


def load_embeddings_cache(
    embs_path: Path = _EMBS_CACHE_PATH,
    ids_path: Path  = _IDS_CACHE_PATH,
) -> tuple[np.ndarray, list[str]] | None:
    """
    Load cached embeddings and IDs from disk.

    Returns
    -------
    (embeddings, candidate_ids) tuple, or None if either file is missing.
    """
    if not embs_path.exists() or not ids_path.exists():
        logger.debug("Cache miss: %s or %s not found.", embs_path, ids_path)
        return None

    try:
        embeddings    = np.load(str(embs_path)).astype(np.float32)
        candidate_ids = np.load(str(ids_path), allow_pickle=True).tolist()
        logger.info(
            "Cache hit: loaded %s embeddings for %d candidates.",
            embeddings.shape, len(candidate_ids),
        )
        return embeddings, candidate_ids
    except Exception as exc:
        logger.warning("Cache load failed (%s); will re-encode.", exc)
        return None


def cache_is_valid(
    embs_path: Path = _EMBS_CACHE_PATH,
    ids_path: Path  = _IDS_CACHE_PATH,
    jsonl_path: Path = CANDIDATES_PATH,
) -> bool:
    """
    Return True when the cache files exist AND are newer than the source JSONL.
    This ensures a stale cache is never silently used after the dataset changes.
    """
    if not embs_path.exists() or not ids_path.exists():
        return False
    if not jsonl_path.exists():
        return True   # can't compare — trust the cache
    cache_mtime = min(embs_path.stat().st_mtime, ids_path.stat().st_mtime)
    data_mtime  = jsonl_path.stat().st_mtime
    return cache_mtime >= data_mtime


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper used by __main__ and the orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_embeddings(
    jsonl_path: Path | None = None,
    force: bool = False,
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Return (embeddings, candidate_ids) for every candidate in the JSONL file.

    Cache logic
    -----------
    If a valid cache exists AND `force=False`, return the cached values
    immediately without loading the model.  Pass `force=True` to always
    re-encode (e.g. after the model or text builder changes).

    Parameters
    ----------
    jsonl_path     : path to candidates.jsonl (defaults to config.CANDIDATES_PATH)
    force          : ignore existing cache and re-encode
    batch_size     : sentences per inference batch
    show_progress  : show tqdm bar

    Returns
    -------
    embeddings    : float32 ndarray (N, D), unit-norm rows
    candidate_ids : list of str, same order as embeddings rows
    """
    # Lazy import here to keep embedder.py independent of candidate_processor
    # when used as a library (embed_texts / embed_jd don't need it).
    from candidate_processor import load_candidates_unified_cached

    src_path = Path(jsonl_path) if jsonl_path else CANDIDATES_PATH

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if not force and cache_is_valid(_EMBS_CACHE_PATH, _IDS_CACHE_PATH, src_path):
        result = load_embeddings_cache()
        if result is not None:
            return result

    # ── Build texts from JSONL stream (single pass, shared with feature step) ──
    logger.info("Streaming candidates from '%s' …", src_path)
    _raw_lookup, ids, texts = load_candidates_unified_cached(src_path)
    logger.info("Collected %d candidate texts.", len(texts))

    if not texts:
        raise RuntimeError("No candidates found in the JSONL file.")

    # ── Encode ────────────────────────────────────────────────────────────────
    embeddings = embed_texts(texts, batch_size=batch_size, show_progress=show_progress)

    if using_fallback_embedder():
        _model.save(_FALLBACK_ENCODER_CACHE_PATH)
        logger.info("Fallback TF-IDF+SVD encoder (fit on %d docs) cached → %s",
                    len(texts), _FALLBACK_ENCODER_CACHE_PATH.name)

    # ── Persist ───────────────────────────────────────────────────────────────
    save_embeddings_cache(embeddings, ids)

    return embeddings, ids


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Embed candidates and build a FAISS index."
    )
    ap.add_argument(
        "--jsonl", type=Path, default=None,
        help="Path to candidates.jsonl (default: config.CANDIDATES_PATH)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Ignore existing embedding cache and re-encode from scratch.",
    )
    ap.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Encoding batch size (default: {BATCH_SIZE})",
    )
    args = ap.parse_args()

    # ── Late import of indexer here avoids circular deps at module level ──────
    try:
        from indexer import build_faiss_index, save_index
    except ImportError as err:
        sys.exit(f"ERROR: Could not import indexer.py — {err}")

    t_start = time.perf_counter()

    try:
        embeddings, candidate_ids = build_candidate_embeddings(
            jsonl_path    = args.jsonl,
            force         = args.force,
            batch_size    = args.batch_size,
            show_progress = True,
        )
    except (FileNotFoundError, RuntimeError) as err:
        sys.exit(f"ERROR: {err}")

    # ── Build and save FAISS index ────────────────────────────────────────────
    index = build_faiss_index(embeddings)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_index(index, FAISS_INDEX_PATH)

    elapsed = time.perf_counter() - t_start
    n = len(candidate_ids)
    print(f"\nIndexed {n:,} candidates  [{elapsed:.1f}s total]")
    print(f"  Embeddings : {_EMBS_CACHE_PATH}")
    print(f"  IDs        : {_IDS_CACHE_PATH}")
    print(f"  FAISS index: {FAISS_INDEX_PATH}")
