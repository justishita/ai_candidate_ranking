"""
app.py — FastAPI entry point for the AI Candidate Ranking System.

Run with:
    uvicorn app:app --reload --port 8000

Endpoints
---------
  GET  /api/health             — liveness + dependency check
  POST /api/run                — trigger a pipeline run (background task)
  GET  /api/status             — poll current run progress
  GET  /api/results            — fetch ranked candidates for the last completed run
  GET  /api/jd                 — fetch the parsed JD profile for the last run
  GET  /api/download           — download submission.csv
  GET  /                       — serves the frontend (frontend/index.html)

The frontend (frontend/index.html) is a static SPA that calls these endpoints
via fetch() instead of using hardcoded sample data.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import OUTPUT_DIR, WEIGHTS
from pipeline_executor import execute_pipeline
from run_state import run_state
from schemas import (
    ErrorResponse,
    HealthResponse,
    JDProfileOut,
    RankedCandidateOut,
    ResultsResponse,
    RunRequest,
    RunStatusResponse,
    SubScores,
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("app")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(
    title       = "AI Candidate Ranking System",
    description = "Ranks candidates against a job description using hybrid semantic + rule-based scoring.",
    version     = "1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Wide-open for hackathon/demo purposes. Tighten allow_origins to the deployed
# frontend's exact origin before any production use.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """
    Liveness probe. Checks that the heavy ML dependencies are importable
    without actually loading the models (cheap check for orchestration tools).
    """
    pipeline_ok = True
    spacy_ok    = True
    faiss_ok    = True

    try:
        import jd_parser, candidate_processor, embedder, indexer, scorer, ranker  # noqa: F401
    except (Exception, SystemExit) as exc:
        # jd_parser.py calls sys.exit() (raises SystemExit, not Exception) when
        # python-docx or spaCy are missing at import time — must be caught
        # explicitly here, or it propagates past this handler and kills the
        # worker thread instead of producing a normal 200 with pipeline_ok=False.
        logger.warning("Pipeline module import check failed: %s", exc)
        pipeline_ok = False

    try:
        import spacy
        spacy.util.get_package_path("en_core_web_sm")
    except Exception:
        spacy_ok = False

    try:
        import faiss  # noqa: F401
    except Exception:
        faiss_ok = False

    from config import EMBEDDING_MODEL

    return HealthResponse(
        status                  = "ok",
        pipeline_modules_loaded = pipeline_ok,
        spacy_model_loaded      = spacy_ok,
        embedding_model_name    = EMBEDDING_MODEL,
        faiss_available         = faiss_ok,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/api/run",
    response_model=RunStatusResponse,
    status_code=202,
    tags=["pipeline"],
    responses={409: {"model": ErrorResponse}},
)
def trigger_run(request: RunRequest, background_tasks: BackgroundTasks) -> RunStatusResponse:
    """
    Trigger a new pipeline run. Runs in a background thread so the request
    returns immediately (202 Accepted) — poll GET /api/status for progress.

    Returns 409 if a run is already in progress; only one run executes at a
    time (see run_state.py design note).
    """
    if run_state.is_busy():
        raise HTTPException(
            status_code = 409,
            detail      = f"A pipeline run ({run_state.run_id}) is already in progress.",
        )

    run_state.start_new_run()

    # Use a raw background thread rather than FastAPI's BackgroundTasks for the
    # actual ML work — BackgroundTasks runs *after* the response is sent but
    # still within the same worker's event loop context, which would block
    # subsequent requests (including status polls) for a CPU-bound 4-minute job.
    thread = threading.Thread(
        target = execute_pipeline,
        kwargs = dict(
            jd_path         = request.jd_path,
            candidates_path = request.candidates_path,
            faiss_k         = request.faiss_k,
            top_n           = request.top_n,
            force_reembed   = request.force_reembed,
            use_lgbm        = request.use_lgbm,
        ),
        daemon = True,
    )
    thread.start()

    logger.info("Pipeline run %s started in background thread.", run_state.run_id)
    return RunStatusResponse(**run_state.to_status_dict())


@app.get("/api/status", response_model=RunStatusResponse, tags=["pipeline"])
def get_status() -> RunStatusResponse:
    """Poll the progress of the current (or most recently completed) run."""
    return RunStatusResponse(**run_state.to_status_dict())


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/api/jd",
    response_model=JDProfileOut,
    tags=["results"],
    responses={404: {"model": ErrorResponse}},
)
def get_jd_profile() -> JDProfileOut:
    """Return the parsed JD profile for the most recent run."""
    status = run_state.to_status_dict()
    if run_state.jd_profile is None:
        raise HTTPException(status_code=404, detail="No JD has been parsed yet. Run the pipeline first.")
    return JDProfileOut(**run_state.jd_profile)


@app.get(
    "/api/results",
    response_model=ResultsResponse,
    tags=["results"],
    responses={404: {"model": ErrorResponse}, 425: {"model": ErrorResponse}},
)
def get_results() -> ResultsResponse:
    """
    Return the full ranked candidate list for the most recently completed run.

    425 (Too Early) if a run is still in progress; 404 if no run has ever
    completed successfully.
    """
    if run_state.is_busy():
        raise HTTPException(
            status_code = 425,
            detail      = "Pipeline is still running — poll /api/status until status is 'completed'.",
        )
    if run_state.status != "completed" or not run_state.ranked:
        raise HTTPException(
            status_code = 404,
            detail      = "No completed run available. POST /api/run first.",
        )

    candidates: list[RankedCandidateOut] = []
    for r in run_state.ranked:
        candidates.append(RankedCandidateOut(
            candidate_id = r["candidate_id"],
            rank         = r["rank"],
            score        = r["score"],
            reasoning    = r.get("reasoning", ""),
            sub_scores   = SubScores(
                semantic_similarity = r["semantic_similarity"],
                skill_match         = r["skill_match"],
                experience_score    = r["experience_score"],
                career_history      = r["career_history"],
                behaviour_score     = r["behaviour_score"],
                education_score     = r["education_score"],
                location_score      = r["location_score"],
                quality_score       = r["quality_score"],
            ),
            name                    = r.get("name"),
            role                    = r.get("role"),
            company                 = r.get("company"),
            location                = r.get("location"),
            remote                  = r.get("remote"),
            total_experience_years  = r.get("total_experience_years"),
            skills                  = r.get("skills", []),
        ))

    avg_score = (
        sum(c.score for c in candidates) / len(candidates) if candidates else 0.0
    )

    jd_profile_dict = run_state.jd_profile or {}

    return ResultsResponse(
        run_id       = run_state.run_id,
        jd_profile   = JDProfileOut(**jd_profile_dict) if jd_profile_dict else JDProfileOut(
            role_title="Unknown", seniority="unknown"
        ),
        weights      = run_state.weights or WEIGHTS,
        candidates   = candidates,
        avg_score    = round(avg_score, 4),
        generated_at = run_state.finished_at or "",
    )


@app.get(
    "/api/download",
    tags=["results"],
    responses={404: {"model": ErrorResponse}},
)
def download_submission() -> FileResponse:
    """Download the generated submission.csv from the most recent run."""
    csv_path = OUTPUT_DIR / "submission.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = "submission.csv not found — run the pipeline first.",
        )
    return FileResponse(
        path        = csv_path,
        media_type  = "text/csv",
        filename    = "submission.csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Frontend (static SPA)
# ─────────────────────────────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("Frontend directory not found at %s — only API routes are served.", FRONTEND_DIR)
