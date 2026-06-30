"""
schemas.py — Pydantic models for API request/response validation.

Keeping these separate from the pipeline modules means the pipeline code
(jd_parser, ranker, etc.) stays framework-agnostic — FastAPI is only ever
imported here and in app.py.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Run lifecycle
# ─────────────────────────────────────────────────────────────────────────────

RunStatus = Literal["idle", "queued", "running", "completed", "failed"]


class RunRequest(BaseModel):
    """Body for POST /api/run — all fields optional, fall back to config.py defaults."""
    jd_path: str | None = Field(
        default=None, description="Override path to job_description.docx"
    )
    candidates_path: str | None = Field(
        default=None, description="Override path to candidates.jsonl"
    )
    faiss_k: int = Field(
        default=2000, ge=10, le=20000,
        description="Number of candidates retrieved from FAISS before re-scoring",
    )
    top_n: int = Field(
        default=100, ge=1, le=500,
        description="Number of candidates in the final ranked output",
    )
    force_reembed: bool = Field(
        default=False, description="Ignore embedding cache and re-encode from scratch"
    )
    use_lgbm: bool = Field(
        default=True, description="Apply LightGBM re-ranking if a model file exists"
    )


class StepStatus(BaseModel):
    """One pipeline step's progress, mirrors main.py's 7-step structure."""
    index: int
    name: str
    status: Literal["idle", "active", "done", "failed"]
    detail: str | None = None
    duration_seconds: float | None = None


class RunStatusResponse(BaseModel):
    """Body for GET /api/status — polled by the frontend during a run."""
    run_id: str
    status: RunStatus
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    steps: list[StepStatus] = Field(default_factory=list)
    error: str | None = None
    result_count: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Job description
# ─────────────────────────────────────────────────────────────────────────────

class ExperienceRangeOut(BaseModel):
    min_years: int
    max_years: int
    raw: str


class JDProfileOut(BaseModel):
    role_title: str
    seniority: str
    experience: ExperienceRangeOut | None = None
    mandatory_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Ranked candidates
# ─────────────────────────────────────────────────────────────────────────────

class SubScores(BaseModel):
    semantic_similarity: float
    skill_match: float
    experience_score: float
    career_history: float
    behaviour_score: float
    education_score: float
    location_score: float
    quality_score: float


class RankedCandidateOut(BaseModel):
    candidate_id: str
    rank: int
    score: float
    reasoning: str
    sub_scores: SubScores
    # Light extras for the UI card (skills, experience, location) — optional
    # because they depend on raw_candidate being available at ranking time.
    name: str | None = None
    role: str | None = None
    company: str | None = None
    location: str | None = None
    remote: bool | None = None
    total_experience_years: float | None = None
    skills: list[str] = Field(default_factory=list)


class ResultsResponse(BaseModel):
    run_id: str
    jd_profile: JDProfileOut
    weights: dict[str, float]
    candidates: list[RankedCandidateOut]
    avg_score: float
    generated_at: str


class ValidationSummary(BaseModel):
    valid: bool
    row_count: int
    checks_passed: list[str]
    checks_failed: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    pipeline_modules_loaded: bool
    spacy_model_loaded: bool
    embedding_model_name: str
    faiss_available: bool

class ErrorResponse(BaseModel):
    detail: str
