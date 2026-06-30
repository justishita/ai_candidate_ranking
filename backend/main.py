"""
main.py — Entry point for the AI Candidate Ranking System.

Usage
-----
  python main.py
  python main.py --jd data/job_description.docx \\
                 --candidates data/candidates.jsonl \\
                 --output output/submission.csv \\
                 --faiss-k 2000 \\
                 --force-reembed

Pipeline (7 steps)
------------------
  [1] Parse JD                   → jd_profile
  [2] Load & process candidates  → features streamed on-demand
  [3] Build / load embeddings    → embeddings.npy (cached)
  [4] Build / load FAISS index   → candidate_index.faiss (cached)
  [5] Search FAISS (top-K)       → semantic candidates pool
  [6] Score + rank               → top-100 ranked list
  [7] Generate reasons + write + validate submission.csv

Runtime budget
--------------
  The pipeline monitors elapsed time and emits a warning if it projects
  exceeding MAX_RUNTIME_SECONDS (270 s by default — < 5 min hard cap).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Early config import (paths / constants) ────────────────────────────────
try:
    from config import (
        CANDIDATES_PATH,
        FAISS_INDEX_PATH,
        FAISS_TOP_K,
        FINAL_TOP_N,
        JD_PATH,
        MAX_RUNTIME_SECONDS,
        OUTPUT_DIR,
        SUBMISSION_PATH,
        WEIGHTS,
    )
except ImportError:
    CANDIDATES_PATH     = Path("data/candidates.jsonl")
    JD_PATH             = Path("data/job_description.docx")
    SUBMISSION_PATH     = Path("output/submission.csv")
    FAISS_INDEX_PATH    = Path("output/candidate_index.faiss")
    OUTPUT_DIR          = Path("output")
    FAISS_TOP_K         = 2000
    FINAL_TOP_N         = 100
    MAX_RUNTIME_SECONDS = 270
    WEIGHTS: dict[str, float] = {
        "semantic": 0.30, "skill_match": 0.20, "experience": 0.15,
        "career_history": 0.10, "behaviour": 0.10, "education": 0.05, "location": 0.10,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(
        level   = level,
        format  = fmt,
        datefmt = "%H:%M:%S",
        handlers= [logging.StreamHandler(sys.stdout)],
    )
    # Quiet noisy third-party loggers
    for noisy in ("sentence_transformers", "transformers", "torch", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Progress printer
# ─────────────────────────────────────────────────────────────────────────────

_STEP_WIDTH = 55

def _step(n: int, total: int, desc: str, t_start: float) -> float:
    """Print a timestamped step header and return the current monotonic time."""
    now      = time.perf_counter()
    elapsed  = now - t_start
    ts       = datetime.now().strftime("%H:%M:%S")
    bar      = "█" * n + "░" * (total - n)
    print(
        f"\n[{ts}]  Step {n}/{total}  {bar}\n"
        f"         {desc:<{_STEP_WIDTH}}  (+{elapsed:5.1f}s elapsed)",
        flush=True,
    )
    return now


def _done(desc: str, t_step: float) -> None:
    """Print a ✓ completion line for the last step."""
    duration = time.perf_counter() - t_step
    print(f"         ✓ {desc:<{_STEP_WIDTH}}  [{duration:.1f}s]", flush=True)


def _warn_budget(t_start: float, budget: float = MAX_RUNTIME_SECONDS) -> None:
    """Emit a warning if elapsed time exceeds 80% of the runtime budget."""
    elapsed = time.perf_counter() - t_start
    if elapsed > budget * 0.80:
        logger.warning(
            "⚠  %.0f s elapsed (%.0f%% of %.0f s budget). "
            "Remaining budget: %.0f s.",
            elapsed, 100 * elapsed / budget, budget, budget - elapsed,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "main.py",
        description = "AI Candidate Ranking System — produces submission.csv",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--jd", type=Path, default=JD_PATH,
        metavar="PATH",
        help="Path to job_description.docx",
    )
    p.add_argument(
        "--candidates", type=Path, default=CANDIDATES_PATH,
        metavar="PATH",
        help="Path to candidates.jsonl",
    )
    p.add_argument(
        "--output", type=Path, default=SUBMISSION_PATH,
        metavar="PATH",
        help="Destination path for submission.csv",
    )
    p.add_argument(
        "--faiss-index", type=Path, default=FAISS_INDEX_PATH,
        metavar="PATH",
        help="Path for saving / loading the FAISS index",
    )
    p.add_argument(
        "--faiss-k", type=int, default=FAISS_TOP_K,
        metavar="K",
        help="Number of candidates retrieved from FAISS before re-scoring",
    )
    p.add_argument(
        "--top-n", type=int, default=FINAL_TOP_N,
        metavar="N",
        help="Number of candidates in the final output (≤ faiss-k)",
    )
    p.add_argument(
        "--force-reembed", action="store_true",
        help="Ignore cached embeddings and re-encode from scratch",
    )
    p.add_argument(
        "--no-lgbm", action="store_true",
        help="Disable LightGBM re-ranking even if a model file is present",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    """
    Execute the full ranking pipeline.

    Returns exit code: 0 = success, 1 = error, 2 = validation failure.
    """
    TOTAL_STEPS = 7
    t_wall      = time.perf_counter()   # wall-clock start

    print(
        f"\n{'═'*65}\n"
        f"  AI Candidate Ranking System\n"
        f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  JD      : {args.jd}\n"
        f"  Data    : {args.candidates}\n"
        f"  Output  : {args.output}\n"
        f"{'═'*65}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Late imports (keep startup fast; heavy libs load here) ────────────────
    from candidate_processor import extract_features, stream_candidates
    from embedder import build_candidate_embeddings, embed_jd
    from indexer import (
        build_faiss_index,
        faiss_results_to_ranked,
        load_index,
        save_index,
        search_index,
    )
    from jd_parser import parse_jd
    from output import validate_output, write_submission_csv
    from ranker import _load_lgbm_model, rank_candidates
    from reason_generator import generate_reasons_bulk

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — Parse JD
    # ──────────────────────────────────────────────────────────────────────────
    t_step = _step(1, TOTAL_STEPS, "Parsing job description …", t_wall)
    try:
        jd_profile: dict[str, Any] = parse_jd(args.jd)
    except FileNotFoundError as exc:
        logger.error("JD file not found: %s", exc)
        return 1
    except Exception as exc:
        logger.error("JD parsing failed: %s", exc, exc_info=True)
        return 1

    _done(
        f"Role: '{jd_profile.get('role_title', '?')}' | "
        f"Seniority: {jd_profile.get('seniority', '?')} | "
        f"Mandatory skills: {len(jd_profile.get('mandatory_skills', []))}",
        t_step,
    )
    _warn_budget(t_wall)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — Build / load candidate embeddings (with cache)
    # ──────────────────────────────────────────────────────────────────────────
    t_step = _step(2, TOTAL_STEPS, "Building candidate embeddings (cached) …", t_wall)
    try:
        embeddings, candidate_ids = build_candidate_embeddings(
            jsonl_path    = args.candidates,
            force         = args.force_reembed,
            show_progress = True,
        )
    except FileNotFoundError as exc:
        logger.error("Candidates file not found: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Embedding build failed: %s", exc, exc_info=True)
        return 1

    _done(f"Encoded {len(candidate_ids):,} candidates → shape {embeddings.shape}", t_step)
    _warn_budget(t_wall)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — Build / load FAISS index
    # ──────────────────────────────────────────────────────────────────────────
    t_step = _step(3, TOTAL_STEPS, "Building / loading FAISS index …", t_wall)
    try:
        if args.faiss_index.exists() and not args.force_reembed:
            index = load_index(args.faiss_index)
            logger.info("Loaded existing FAISS index from '%s'.", args.faiss_index)
        else:
            index = build_faiss_index(embeddings)
            save_index(index, args.faiss_index)
    except Exception as exc:
        logger.error("FAISS index build/load failed: %s", exc, exc_info=True)
        return 1

    _done(f"Index ready: {index.ntotal:,} vectors, dim={index.d}", t_step)
    _warn_budget(t_wall)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4 — Embed JD and search FAISS
    # ──────────────────────────────────────────────────────────────────────────
    t_step = _step(4, TOTAL_STEPS, f"Searching FAISS (top-{args.faiss_k}) …", t_wall)
    try:
        jd_vec             = embed_jd(jd_profile)
        distances, indices = search_index(index, jd_vec, k=args.faiss_k)
        faiss_results      = faiss_results_to_ranked(distances, indices, candidate_ids)
    except Exception as exc:
        logger.error("FAISS search failed: %s", exc, exc_info=True)
        return 1

    top_sem_score = faiss_results[0]["semantic_score"] if faiss_results else 0.0
    _done(
        f"Retrieved {len(faiss_results):,} candidates  "
        f"(top semantic score: {top_sem_score:.4f})",
        t_step,
    )
    _warn_budget(t_wall)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5 — Extract features for FAISS pool (stream, only retrieved IDs)
    # ──────────────────────────────────────────────────────────────────────────
    t_step = _step(
        5, TOTAL_STEPS,
        f"Extracting features for {len(faiss_results):,} FAISS candidates …",
        t_wall,
    )
    retrieved_ids = {r["candidate_id"] for r in faiss_results}
    features_lookup: dict[str, dict[str, Any]] = {}
    raw_lookup:      dict[str, dict[str, Any]] = {}

    try:
        for cand in stream_candidates(args.candidates):
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
    except Exception as exc:
        logger.error("Feature extraction failed: %s", exc, exc_info=True)
        return 1

    _done(f"Features ready for {len(features_lookup):,} candidates", t_step)
    _warn_budget(t_wall)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6 — Score, (LightGBM re-rank), return top-N
    # ──────────────────────────────────────────────────────────────────────────
    t_step = _step(6, TOTAL_STEPS, "Scoring and ranking candidates …", t_wall)

    lgbm_model = None
    if not args.no_lgbm:
        lgbm_model = _load_lgbm_model()
        if lgbm_model:
            logger.info("LightGBM re-ranker will be applied.")
        else:
            logger.info("LightGBM model not found — using hybrid score only.")

    try:
        ranked: list[dict[str, Any]] = rank_candidates(
            faiss_results         = faiss_results,
            features_lookup       = features_lookup,
            jd_profile            = jd_profile,
            weights               = WEIGHTS,
            top_n                 = args.top_n,
            lgbm_model            = lgbm_model,
            raw_candidates_lookup = raw_lookup,
        )
    except Exception as exc:
        logger.error("Ranking failed: %s", exc, exc_info=True)
        return 1

    top_score = ranked[0]["score"] if ranked else 0.0
    _done(
        f"Top-{len(ranked)} selected  "
        f"(#1 score: {top_score:.4f}  id: {ranked[0]['candidate_id'] if ranked else '—'})",
        t_step,
    )
    _warn_budget(t_wall)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 7 — Generate reasons, write CSV, validate
    # ──────────────────────────────────────────────────────────────────────────
    t_step = _step(7, TOTAL_STEPS, "Generating reasons & writing submission.csv …", t_wall)

    try:
        reasons = generate_reasons_bulk(ranked, jd_profile, raw_lookup)
    except Exception as exc:
        logger.warning("Reason generation failed (%s) — using fallback strings.", exc)
        reasons = [f"Score: {r['score']:.4f}" for r in ranked]

    try:
        out_path = write_submission_csv(
            ranked_candidates = ranked,
            reasons           = reasons,
            output_path       = args.output,
            top_n             = args.top_n,
        )
    except Exception as exc:
        logger.error("CSV write failed: %s", exc, exc_info=True)
        return 1

    # ── Validate ──────────────────────────────────────────────────────────────
    try:
        validation = validate_output(out_path, expected_rows=args.top_n)
    except Exception as exc:
        logger.error("Validation raised an exception: %s", exc, exc_info=True)
        validation = {"valid": False, "checks_failed": [str(exc)], "row_count": -1}

    _done(
        f"Written {validation.get('row_count', '?')} rows → {out_path.name}  "
        f"{'✓ VALID' if validation['valid'] else '✗ INVALID'}",
        t_step,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_wall

    print(f"\n{'═'*65}")
    print(f"  Pipeline complete in {total_elapsed:.1f}s  "
          f"({'WITHIN' if total_elapsed < MAX_RUNTIME_SECONDS else 'OVER'} budget)")
    print(f"  Output  : {out_path}")
    print(f"  Rows    : {validation.get('row_count', '?')}")
    print(f"  Valid   : {'YES ✓' if validation['valid'] else 'NO ✗'}")

    if not validation["valid"]:
        print(f"\n  Failed checks:")
        for msg in validation.get("checks_failed", []):
            print(f"    ✗  {msg}")
        print(f"{'═'*65}\n")
        return 2

    print(f"\n  Top-5 preview:")
    print(f"  {'Rank':>4}  {'Candidate ID':<24}  {'Score':>6}  Reasoning[:60]")
    print(f"  {'─'*4}  {'─'*24}  {'─'*6}  {'─'*60}")
    for r, reason in zip(ranked[:5], reasons[:5]):
        print(
            f"  {r['rank']:>4}  {str(r['candidate_id']):<24}  "
            f"{r['score']:>6.4f}  {reason[:60]}"
        )
    print(f"{'═'*65}\n")

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    _setup_logging(verbose=_args.verbose)
    sys.exit(run(_args))
