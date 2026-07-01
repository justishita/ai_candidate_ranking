"""
ranker.py — Re-ranking layer for the AI Candidate Ranking System.

Pipeline
--------
1. Receive top-K FAISS candidates (candidate_id + semantic_score + features).
2. Compute hybrid_score for every candidate via scorer.py.
3. Optionally apply a LightGBM re-ranker if a saved model exists (skip gracefully).
4. Sort descending, assign ranks 1-N, return top-100 as list of dicts.

__main__ block
--------------
  python ranker.py [--jsonl path] [--jd path] [--top N] [--no-lgbm]

  Runs the full pipeline end-to-end on real (or synthetic) data and prints
  the top-10 results in a formatted table.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from config import (
        CANDIDATES_PATH,
        FAISS_INDEX_PATH,
        FAISS_TOP_K,
        FINAL_TOP_N,
        JD_PATH,
        LIGHTGBM_MODEL_PATH,
        WEIGHTS,
    )
except ImportError:
    CANDIDATES_PATH      = Path("data/candidates.jsonl")
    FAISS_INDEX_PATH     = Path("output/candidate_index.faiss")
    FAISS_TOP_K          = 500
    FINAL_TOP_N          = 100
    JD_PATH              = Path("data/job_description.docx")
    LIGHTGBM_MODEL_PATH  = Path("models/ranker.lgb")
    WEIGHTS: dict[str, float] = {
        "semantic": 0.30, "skill_match": 0.20, "experience": 0.15,
        "career_history": 0.10, "behaviour": 0.10, "education": 0.05, "location": 0.10,
    }

from scorer import compute_all_scores, hybrid_score

logger = logging.getLogger(__name__)

# Feature columns fed to LightGBM (must match training schema if model exists)
_LGBM_FEATURE_COLS: list[str] = [
    "semantic_similarity",
    "skill_match",
    "experience_score",
    "career_history",
    "behaviour_score",
    "education_score",
    "location_score",
    "quality_score",
    # Raw features from candidate_processor
    "total_experience_years",
    "relevant_experience_years",
    "skill_match_score",        # redundant but keeps schema stable
]


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM loader (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _load_lgbm_model(model_path: Path = LIGHTGBM_MODEL_PATH) -> Any | None:
    """
    Attempt to load a saved LightGBM Booster from disk.

    Returns the Booster object on success, None if the file doesn't exist or
    lightgbm is not installed.  The caller must handle the None case.
    """
    if not model_path.exists():
        logger.info(
            "LightGBM model not found at '%s' — skipping re-ranking.", model_path
        )
        return None

    try:
        import lightgbm as lgb  # noqa: PLC0415
        model = lgb.Booster(model_file=str(model_path))
        logger.info("LightGBM re-ranker loaded from '%s'.", model_path)
        return model
    except ImportError:
        logger.warning("lightgbm package not installed — skipping LightGBM re-ranking.")
        return None
    except Exception as exc:
        logger.warning("Failed to load LightGBM model: %s — skipping.", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core ranking function
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(
    faiss_results: list[dict[str, Any]],
    features_lookup: dict[str, dict[str, Any]],
    jd_profile: dict[str, Any],
    weights: dict[str, float] | None = None,
    top_n: int = FINAL_TOP_N,
    lgbm_model: Any | None = None,
    raw_candidates_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Re-rank FAISS candidates and return the top-N as structured dicts.

    Parameters
    ----------
    faiss_results         : list of {candidate_id, semantic_score, faiss_index}
                            from indexer.faiss_results_to_ranked
    features_lookup       : {candidate_id → feature_dict} from candidate_processor
    jd_profile            : JDProfile dict from jd_parser.parse_jd
    weights               : scoring weight dict (defaults to config.WEIGHTS)
    top_n                 : how many ranked results to return
    lgbm_model            : loaded LightGBM Booster or None
    raw_candidates_lookup : optional {candidate_id → raw JSONL dict} for career
                            history scoring; pass None to skip that signal

    Returns
    -------
    List of up to top_n dicts, each containing:
      candidate_id, rank, score, semantic_similarity, skill_match,
      experience_score, career_history, behaviour_score, education_score,
      location_score, quality_score
    """
    w = weights if weights is not None else WEIGHTS

    if not faiss_results:
        logger.warning("rank_candidates: faiss_results is empty.")
        return []

    rows: list[dict[str, Any]] = []

    for item in faiss_results:
        cid          = str(item["candidate_id"])
        sem_score    = float(item.get("semantic_score", 0.0))
        features     = features_lookup.get(cid, {})
        raw_cand     = (raw_candidates_lookup or {}).get(cid)

        try:
            sub_scores = compute_all_scores(
                candidate_features = features,
                jd_profile         = jd_profile,
                semantic_score     = sem_score,
                raw_candidate      = raw_cand,
            )
        except Exception as exc:
            logger.warning("compute_all_scores failed for %s: %s", cid, exc)
            sub_scores = {
                "semantic_similarity": 0.0, "skill_match": 0.0,
                "experience_score": 0.0,    "career_history": 0.0,
                "behaviour_score": 0.5,     "education_score": 0.5,
                "location_score": 0.0,      "quality_score": 1.0,
            }

        try:
            h_score = round(hybrid_score(sub_scores, w), 4)
        except Exception as exc:
            logger.warning("hybrid_score failed for %s: %s", cid, exc)
            h_score = 0.0

        row = {
            "candidate_id"      : cid,
            "hybrid_score"      : h_score,
            # Extra raw features carried for LightGBM feature matrix
            "total_experience_years"    : float(features.get("total_experience_years", 0.0)),
            "relevant_experience_years" : float(features.get("relevant_experience_years", 0.0)),
            "skill_match_score"         : float(features.get("skill_match_score", 0.0)),
            "skills_list"               : features.get("skills_list", []),
            **sub_scores,
        }
        rows.append(row)

    # ── Stage 1: sort by hybrid score, tie-broken by candidate_id ascending ───
    # (submission_spec.md §3: "If two candidates have the same score, you must
    # still assign unique ranks... by candidate_id ascending.")
    rows.sort(key=lambda r: r["candidate_id"])
    rows.sort(key=lambda r: r["hybrid_score"], reverse=True)

    # ── Stage 2: optional LightGBM re-ranking ────────────────────────────────
    if lgbm_model is not None:
        rows = _lgbm_rerank(rows, lgbm_model, top_n)
    else:
        rows = rows[:top_n]

    # ── Assign ranks and clean up internal columns ─────────────────────────────
    ranked: list[dict[str, Any]] = []
    for rank_idx, row in enumerate(rows, start=1):
        ranked.append({
            "candidate_id"       : row["candidate_id"],
            "rank"               : rank_idx,
            "score"              : row.get("lgbm_score", row["hybrid_score"]),
            "semantic_similarity": row["semantic_similarity"],
            "skill_match"        : row["skill_match"],
            "experience_score"   : row["experience_score"],
            "career_history"     : row["career_history"],
            "behaviour_score"    : row["behaviour_score"],
            "education_score"    : row["education_score"],
            "location_score"     : row["location_score"],
            "quality_score"      : row["quality_score"],
            "hybrid_score"       : row["hybrid_score"],
            # Carried through for reason_generator — the raw JSONL candidate
            # has none of these as flat fields (years_of_experience lives at
            # profile.years_of_experience, skills is a list of dicts, etc.),
            # so reasoning generation needs the already-computed feature
            # values, not the raw record.
            "total_experience_years"    : row.get("total_experience_years", 0.0),
            "relevant_experience_years" : row.get("relevant_experience_years", 0.0),
            "skill_match_score"         : row.get("skill_match_score", 0.0),
            "skills_list"               : row.get("skills_list", []),
        })

    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM re-ranking helper
# ─────────────────────────────────────────────────────────────────────────────

def _lgbm_rerank(
    rows: list[dict[str, Any]],
    model: Any,
    top_n: int,
) -> list[dict[str, Any]]:
    """
    Run LightGBM inference over the candidate feature matrix, add a
    ``lgbm_score`` field, and return rows sorted by that score (top_n only).

    The model is expected to output a relevance score (higher = better fit).
    If inference fails for any reason, we fall back to the hybrid_score order.
    """
    try:
        feature_matrix = pd.DataFrame(rows)[_LGBM_FEATURE_COLS].fillna(0.0)
        lgbm_scores: np.ndarray = model.predict(feature_matrix.values)

        for row, score in zip(rows, lgbm_scores):
            row["lgbm_score"] = float(score)

        rows.sort(key=lambda r: r["candidate_id"])
        rows.sort(key=lambda r: r["lgbm_score"], reverse=True)
        logger.info("LightGBM re-ranking applied to %d candidates.", len(rows))
    except Exception as exc:
        logger.warning("LightGBM inference failed (%s) — using hybrid_score order.", exc)

    return rows[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience end-to-end runner (used by pipeline.py and __main__)
# ─────────────────────────────────────────────────────────────────────────────

def run_ranking_pipeline(
    jsonl_path: Path | None = None,
    jd_path: Path | None = None,
    faiss_path: Path | None = None,
    top_n: int = FINAL_TOP_N,
    faiss_k: int = FAISS_TOP_K,
    use_lgbm: bool = True,
    force_reembed: bool = False,
) -> list[dict[str, Any]]:
    """
    Full end-to-end ranking: parse JD → embed candidates → FAISS search →
    score → (LightGBM) → return top-N ranked list.

    Parameters
    ----------
    jsonl_path    : path to candidates.jsonl
    jd_path       : path to job_description.docx
    faiss_path    : path to saved FAISS index (built if missing)
    top_n         : number of candidates in the final output
    faiss_k       : number of candidates retrieved from FAISS before re-scoring
    use_lgbm      : whether to attempt LightGBM re-ranking
    force_reembed : pass True to ignore the embedding cache

    Returns
    -------
    Ranked list as returned by rank_candidates.
    """
    from candidate_processor import extract_features, stream_candidates
    from embedder import build_candidate_embeddings, embed_jd
    from indexer import build_faiss_index, faiss_results_to_ranked, load_index, save_index
    from jd_parser import parse_jd

    src  = Path(jsonl_path)  if jsonl_path else CANDIDATES_PATH
    jd   = Path(jd_path)     if jd_path    else JD_PATH
    fidx = Path(faiss_path)  if faiss_path else FAISS_INDEX_PATH

    t0 = time.perf_counter()

    # ── 1. Parse JD ──────────────────────────────────────────────────────────
    logger.info("[1/5] Parsing JD …")
    jd_profile = parse_jd(jd)

    # ── 2. Build / load candidate embeddings ─────────────────────────────────
    logger.info("[2/5] Building candidate embeddings …")
    embeddings, candidate_ids = build_candidate_embeddings(
        jsonl_path=src, force=force_reembed
    )

    # ── 3. Build / load FAISS index ──────────────────────────────────────────
    logger.info("[3/5] Building FAISS index …")
    if fidx.exists() and not force_reembed:
        index = load_index(fidx)
    else:
        index = build_faiss_index(embeddings)
        save_index(index, fidx)

    # ── 4. Embed JD and search ───────────────────────────────────────────────
    logger.info("[4/5] Searching FAISS index …")
    jd_vec            = embed_jd(jd_profile)
    from indexer import search_index
    distances, indices = search_index(index, jd_vec, k=faiss_k)
    faiss_results      = faiss_results_to_ranked(distances, indices, candidate_ids)

    # ── 5. Extract features for retrieved candidates ─────────────────────────
    logger.info("[5/5] Scoring and ranking …")
    retrieved_ids = {r["candidate_id"] for r in faiss_results}
    features_lookup: dict[str, dict[str, Any]] = {}
    raw_lookup:      dict[str, dict[str, Any]] = {}

    for cand in stream_candidates(src):
        cid = str(
            cand.get("candidate_id") or cand.get("id") or cand.get("_id") or ""
        )
        if cid in retrieved_ids:
            features_lookup[cid] = extract_features(
                cand,
                jd_mandatory_skills = jd_profile.get("mandatory_skills"),
                jd_locations        = jd_profile.get("locations"),
            )
            raw_lookup[cid] = cand

    # ── 6. Load LightGBM if requested ────────────────────────────────────────
    lgbm_model = _load_lgbm_model() if use_lgbm else None

    # ── 7. Rank ──────────────────────────────────────────────────────────────
    ranked = rank_candidates(
        faiss_results         = faiss_results,
        features_lookup       = features_lookup,
        jd_profile            = jd_profile,
        top_n                 = top_n,
        lgbm_model            = lgbm_model,
        raw_candidates_lookup = raw_lookup,
    )

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline complete: %d candidates ranked in %.1f s.", len(ranked), elapsed)
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# CLI / __main__
# ─────────────────────────────────────────────────────────────────────────────

def _print_top(ranked: list[dict[str, Any]], n: int = 10) -> None:
    """Pretty-print the top-N results as a fixed-width table."""
    header = (
        f"{'Rank':>4}  {'Candidate ID':<24}  {'Score':>6}  "
        f"{'Sem':>5}  {'Skill':>5}  {'Exp':>5}  "
        f"{'Career':>6}  {'Beh':>5}  {'Edu':>5}  {'Loc':>5}  {'Qual':>5}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in ranked[:n]:
        print(
            f"{r['rank']:>4}  {str(r['candidate_id']):<24}  "
            f"{r['score']:>6.4f}  "
            f"{r['semantic_similarity']:>5.3f}  "
            f"{r['skill_match']:>5.3f}  "
            f"{r['experience_score']:>5.3f}  "
            f"{r['career_history']:>6.3f}  "
            f"{r['behaviour_score']:>5.3f}  "
            f"{r['education_score']:>5.3f}  "
            f"{r['location_score']:>5.3f}  "
            f"{r['quality_score']:>5.3f}"
        )
    print(sep)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="Run the full candidate ranking pipeline.")
    ap.add_argument("--jsonl",        type=Path, default=None,
                    help="Path to candidates.jsonl")
    ap.add_argument("--jd",           type=Path, default=None,
                    help="Path to job_description.docx")
    ap.add_argument("--faiss",        type=Path, default=None,
                    help="Path to FAISS index file")
    ap.add_argument("--top",          type=int,  default=FINAL_TOP_N,
                    help=f"Return top-N candidates (default: {FINAL_TOP_N})")
    ap.add_argument("--faiss-k",      type=int,  default=FAISS_TOP_K,
                    help=f"FAISS retrieval pool size (default: {FAISS_TOP_K})")
    ap.add_argument("--no-lgbm",      action="store_true",
                    help="Disable LightGBM re-ranking even if model exists")
    ap.add_argument("--force-reembed",action="store_true",
                    help="Ignore embedding cache and re-encode from scratch")
    ap.add_argument("--print-top",    type=int,  default=10,
                    help="Number of top results to print (default: 10)")
    ap.add_argument("--synthetic",    action="store_true",
                    help="Run on synthetic data (no real files needed — for CI/smoke test)")
    args = ap.parse_args()

    # ── Synthetic smoke-test mode ─────────────────────────────────────────────
    if args.synthetic:
        logger.info("Running synthetic smoke-test …")
        import random
        random.seed(42)

        # Fake JD profile
        fake_jd: dict[str, Any] = {
            "role_title"      : "Senior ML Engineer",
            "seniority"       : "senior",
            "experience"      : {"min_years": 5, "max_years": 10, "raw": "5-10 years"},
            "mandatory_skills": ["python", "pytorch", "aws", "docker"],
            "preferred_skills": ["kubernetes", "mlflow"],
            "locations"       : ["Bangalore", "Remote"],
            "raw_text"        : "Senior ML Engineer with 5-10 years experience in python pytorch aws docker",
        }

        # Fake FAISS results
        n_fake = 50
        fake_ids = [f"cand_{i:04d}" for i in range(n_fake)]
        fake_faiss = [
            {
                "candidate_id"  : cid,
                "semantic_score": random.uniform(0.55, 0.99),
                "faiss_index"   : idx,
            }
            for idx, cid in enumerate(fake_ids)
        ]

        # Fake feature dicts
        degrees     = ["BTech", "Masters", "PhD", "Diploma"]
        loc_matches = [0.0, 0.5, 0.8, 1.0]
        fake_features: dict[str, dict[str, Any]] = {}
        for cid in fake_ids:
            fake_features[cid] = {
                "candidate_id"              : cid,
                "total_experience_years"    : round(random.uniform(1, 15), 1),
                "relevant_experience_years" : round(random.uniform(0, 10), 1),
                "skills_list"               : random.sample(
                    ["python", "pytorch", "aws", "docker", "kafka", "spark", "go"], k=4
                ),
                "skill_match_score"         : round(random.uniform(0, 1), 3),
                "education_score"           : random.choice([0.5, 0.7, 0.85, 1.0]),
                "location_match"            : random.choice(loc_matches),
                "behaviour_score"           : round(random.uniform(0.3, 1.0), 3),
                "profile_quality_score"     : round(random.uniform(0.5, 1.0), 3),
            }

        # Fake raw candidates (minimal, for career history)
        fake_raw: dict[str, dict[str, Any]] = {}
        for cid in fake_ids:
            n_roles = random.randint(1, 5)
            fake_raw[cid] = {
                "work_experience": [
                    {
                        "title"     : random.choice([
                            "Junior Engineer", "Software Engineer",
                            "Senior Engineer", "Tech Lead", "Engineering Manager",
                        ]),
                        "company"   : random.choice([
                            "Google", "StartupXYZ", "Infosys", "Amazon", "Unknown Corp"
                        ]),
                        "start_date": f"{2024 - n_roles + i}-01-01",
                        "end_date"  : f"{2024 - n_roles + i + 1}-01-01",
                    }
                    for i in range(n_roles)
                ]
            }

        lgbm = None if args.no_lgbm else _load_lgbm_model()

        ranked = rank_candidates(
            faiss_results         = fake_faiss,
            features_lookup       = fake_features,
            jd_profile            = fake_jd,
            top_n                 = args.top,
            lgbm_model            = lgbm,
            raw_candidates_lookup = fake_raw,
        )

        print(f"\n[Synthetic] Ranked {len(ranked)} candidates.")
        _print_top(ranked, n=args.print_top)
        sys.exit(0)

    # ── Real pipeline ─────────────────────────────────────────────────────────
    try:
        ranked = run_ranking_pipeline(
            jsonl_path    = args.jsonl,
            jd_path       = args.jd,
            faiss_path    = args.faiss,
            top_n         = args.top,
            faiss_k       = args.faiss_k,
            use_lgbm      = not args.no_lgbm,
            force_reembed = args.force_reembed,
        )
    except (FileNotFoundError, RuntimeError) as err:
        sys.exit(f"ERROR: {err}")

    print(f"\nRanked {len(ranked)} candidates.")
    _print_top(ranked, n=args.print_top)

    # Optionally dump full JSON
    if args.top <= 20:
        print("\nFull JSON output:")
        print(json.dumps(ranked, indent=2))
