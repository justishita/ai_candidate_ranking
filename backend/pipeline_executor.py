"""
pipeline_executor.py — Runs the real candidate-ranking pipeline in a background
thread and reports per-step progress into run_state.run_state.

This module imports directly from the existing pipeline files (jd_parser,
candidate_processor, embedder, indexer, scorer, ranker, reason_generator,
output) — no logic is duplicated. It only adds instrumentation: each of the
7 steps from main.py is wrapped with run_state.step(i).start()/.finish()/.fail()
calls so the frontend's polling loop sees live progress.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

from config import (
    CANDIDATES_PATH,
    FAISS_INDEX_PATH,
    JD_PATH,
    OUTPUT_DIR,
    WEIGHTS,
)
from run_state import run_state

logger = logging.getLogger("pipeline_executor")


def execute_pipeline(
    jd_path: Path | str | None = None,
    candidates_path: Path | str | None = None,
    faiss_k: int = 300,
    top_n: int = 100,
    force_reembed: bool = False,
    use_lgbm: bool = False,
) -> None:
    """
    Run the full ranking pipeline, updating run_state at each of the 7 steps.

    This function is intended to be invoked inside a background thread
    (see app.py's POST /api/run handler) — it writes results into the
    module-level run_state singleton rather than returning a value, since
    the caller (a FastAPI BackgroundTasks worker) has already returned a
    202 Accepted response to the client by the time this runs.
    """
    jd      = Path(jd_path) if jd_path else JD_PATH
    src     = Path(candidates_path) if candidates_path else CANDIDATES_PATH
    fidx    = FAISS_INDEX_PATH
    out_csv = OUTPUT_DIR / "submission.csv"

    run_state.mark_running()
    run_state.set_weights(WEIGHTS)

    try:
        # Late imports: keep API startup fast, heavy ML libs load only when a
        # run is actually triggered. Imports live inside this try block
        # because jd_parser.py raises SystemExit (not Exception) on a missing
        # python-docx/spaCy dependency — that must be caught the same way as
        # any other pipeline failure so the background thread reports it via
        # run_state.fail() instead of dying silently.
        from candidate_processor import extract_features, load_candidates_unified_cached
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

        # ── Step 1 — Parse JD ────────────────────────────────────────────────
        step = run_state.step(1)
        step.start("Reading job_description.docx …")
        jd_profile: dict[str, Any] = parse_jd(jd)
        run_state.set_jd_profile(jd_profile)
        step.finish(
            f"Role: {jd_profile.get('role_title', '?')} · "
            f"{len(jd_profile.get('mandatory_skills', []))} mandatory skills"
        )

        # ── Step 2 — Build / load candidate embeddings ──────────────────────
        step = run_state.step(2)
        step.start("Encoding candidate profiles with BGE …")
        embeddings, candidate_ids = build_candidate_embeddings(
            jsonl_path    = src,
            force         = force_reembed,
            show_progress = False,
        )
        step.finish(f"{len(candidate_ids):,} candidates encoded → shape {embeddings.shape}")

        # ── Step 3 — Build / load FAISS index ───────────────────────────────
        step = run_state.step(3)
        step.start("Indexing embeddings …")
        if fidx.exists() and not force_reembed:
            index = load_index(fidx)
        else:
            index = build_faiss_index(embeddings)
            save_index(index, fidx)
        step.finish(f"{index.ntotal:,} vectors indexed, dim={index.d}")

        # ── Step 4 — Search FAISS ────────────────────────────────────────────
        step = run_state.step(4)
        step.start(f"Retrieving top-{faiss_k} candidates …")
        jd_vec              = embed_jd(jd_profile)
        distances, indices  = search_index(index, jd_vec, k=faiss_k)
        faiss_results        = faiss_results_to_ranked(distances, indices, candidate_ids)
        top_score = faiss_results[0]["semantic_score"] if faiss_results else 0.0
        step.finish(f"{len(faiss_results):,} candidates retrieved · top score {top_score:.4f}")

        # ── Step 5 — Extract features for the FAISS pool ─────────────────────
        step = run_state.step(5)
        step.start(f"Extracting features for {len(faiss_results):,} candidates …")
        retrieved_ids = {r["candidate_id"] for r in faiss_results}
        features_lookup: dict[str, dict[str, Any]] = {}
        raw_lookup:      dict[str, dict[str, Any]] = {}

        all_candidates, _all_ids, _all_texts = load_candidates_unified_cached(src)
        text_by_id = dict(zip(_all_ids, _all_texts))
        for cid in retrieved_ids:
            cand = all_candidates.get(cid)
            if cand is None:
                continue
            features_lookup[cid] = extract_features(
                cand,
                jd_mandatory_skills = jd_profile.get("mandatory_skills"),
                jd_locations        = jd_profile.get("locations"),
                precomputed_embed_text = text_by_id.get(cid),
            )
            raw_lookup[cid] = cand
        step.finish(f"Features ready for {len(features_lookup):,} candidates")

        # ── Step 6 — Score + rank ─────────────────────────────────────────────
        step = run_state.step(6)
        step.start("Computing hybrid scores …")
        lgbm_model = _load_lgbm_model() if use_lgbm else None
        ranked: list[dict[str, Any]] = rank_candidates(
            faiss_results         = faiss_results,
            features_lookup       = features_lookup,
            jd_profile            = jd_profile,
            weights               = WEIGHTS,
            top_n                 = top_n,
            lgbm_model            = lgbm_model,
            raw_candidates_lookup = raw_lookup,
        )
        top1 = ranked[0]["score"] if ranked else 0.0
        step.finish(
            f"Top-{len(ranked)} selected"
            + (f" · #1 score {top1:.4f}" if ranked else "")
            + (" · LightGBM applied" if lgbm_model else " · hybrid score only")
        )

        # ── Step 7 — Reasons + CSV + validation ───────────────────────────────
        step = run_state.step(7)
        step.start("Generating reasoning strings …")
        reasons = generate_reasons_bulk(ranked, jd_profile, raw_lookup)

        out_path = write_submission_csv(
            ranked_candidates = ranked,
            reasons           = reasons,
            output_path       = out_csv,
            top_n             = top_n,
        )
        validation = validate_output(out_path, expected_rows=top_n)
        step.finish(
            f"Written {validation.get('row_count', '?')} rows · "
            f"{'valid' if validation['valid'] else 'INVALID — see checks_failed'}"
        )

        # ── Attach the enriched fields the UI needs onto each ranked dict ─────
        for r, reason in zip(ranked, reasons):
            cid       = r["candidate_id"]
            raw_cand  = raw_lookup.get(cid, {})
            features  = features_lookup.get(cid, {})

            profile = raw_cand.get("profile") or {}

            r["reasoning"]               = reason
            r["name"]                    = (
                raw_cand.get("name") or raw_cand.get("full_name") or
                profile.get("anonymized_name") or cid
            )
            r["company"]                 = (
                _latest_company(raw_cand) or profile.get("current_company") or ""
            )
            r["role"]                    = (
                _latest_title(raw_cand) or profile.get("current_title") or "Unknown role"
            )
            r["location"]                = (
                profile.get("location") or raw_cand.get("location") or
                raw_cand.get("city") or raw_cand.get("current_location") or "Unknown"
            )
            r["remote"]                  = bool(
                raw_cand.get("remote_ok") or raw_cand.get("open_to_remote")
            )
            r["total_experience_years"]  = features.get("total_experience_years", 0.0)
            skills_raw                   = features.get("skills_list", [])
            r["skills"]                  = (
                skills_raw.split("|") if isinstance(skills_raw, str) else skills_raw
            )[:8]

        run_state.complete(ranked, validation)
        logger.info("Pipeline run %s completed: %d candidates.", run_state.run_id, len(ranked))

    except (Exception, SystemExit) as exc:
        tb = traceback.format_exc()
        logger.error("Pipeline run %s failed: %s\n%s", run_state.run_id, exc, tb)
        # Mark whichever step was active as failed
        for s in run_state.steps:
            if s.status == "active":
                s.fail(str(exc))
                break
        run_state.fail(str(exc))


def _latest_title(raw_candidate: dict[str, Any]) -> str:
    """Best-effort extraction of the candidate's most recent job title."""
    profile = raw_candidate.get("profile") or {}
    if profile.get("current_title"):
        return profile["current_title"]
    exps = raw_candidate.get("career_history") or raw_candidate.get("work_experience") or []
    if not exps:
        return profile.get("headline") or raw_candidate.get("headline") or "Unknown role"
    try:
        latest = sorted(
            exps,
            key=lambda e: str(e.get("start_date") or e.get("from") or ""),
            reverse=True,
        )[0]
        return latest.get("title") or latest.get("role") or "Unknown role"
    except Exception:
        return "Unknown role"


def _latest_company(raw_candidate: dict[str, Any]) -> str:
    """Best-effort extraction of the candidate's most recent employer."""
    profile = raw_candidate.get("profile") or {}
    if profile.get("current_company"):
        return profile["current_company"]
    exps = raw_candidate.get("career_history") or raw_candidate.get("work_experience") or []
    if not exps:
        return ""
    try:
        latest = sorted(
            exps,
            key=lambda e: str(e.get("start_date") or e.get("from") or ""),
            reverse=True,
        )[0]
        return latest.get("company") or latest.get("employer") or ""
    except Exception:
        return ""
