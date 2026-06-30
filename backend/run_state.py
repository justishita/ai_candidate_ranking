"""
run_state.py — Thread-safe in-memory state for a single active pipeline run.

Design
------
This hackathon API runs one ranking job at a time (a 100k-candidate run is
CPU/memory heavy enough that concurrent runs aren't a sane default). State
lives in a module-level singleton guarded by a lock so the FastAPI background
thread (executing the pipeline) and the request-handling thread (serving
GET /api/status polls) never race on partial updates.

If you need multi-tenant / concurrent runs later, swap RunState for a
dict[run_id, RunState] and add a run_id parameter throughout — the shape
of each individual run's bookkeeping won't change.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

PIPELINE_STEP_NAMES: list[str] = [
    "Parse job description",
    "Build candidate embeddings",
    "Build / load FAISS index",
    "Search FAISS index",
    "Extract candidate features",
    "Score and rank candidates",
    "Generate reasons & write CSV",
]


class _Step:
    __slots__ = ("index", "name", "status", "detail", "duration_seconds", "_t_start")

    def __init__(self, index: int, name: str) -> None:
        self.index = index
        self.name = name
        self.status: str = "idle"          # idle | active | done | failed
        self.detail: str | None = None
        self.duration_seconds: float | None = None
        self._t_start: float | None = None

    def start(self, detail: str | None = None) -> None:
        self.status = "active"
        self.detail = detail
        self._t_start = time.perf_counter()

    def finish(self, detail: str | None = None) -> None:
        self.status = "done"
        if detail is not None:
            self.detail = detail
        if self._t_start is not None:
            self.duration_seconds = round(time.perf_counter() - self._t_start, 2)

    def fail(self, detail: str) -> None:
        self.status = "failed"
        self.detail = detail
        if self._t_start is not None:
            self.duration_seconds = round(time.perf_counter() - self._t_start, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index"            : self.index,
            "name"             : self.name,
            "status"           : self.status,
            "detail"           : self.detail,
            "duration_seconds" : self.duration_seconds,
        }


class RunState:
    """Mutable state for the currently active (or most recent) pipeline run."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reset_locked()

    # ── Internal helpers (caller must hold _lock) ────────────────────────────

    def _reset_locked(self) -> None:
        self.run_id: str = str(uuid.uuid4())
        self.status: str = "idle"           # idle | queued | running | completed | failed
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.error: str | None = None
        self.steps: list[_Step] = [
            _Step(i, name) for i, name in enumerate(PIPELINE_STEP_NAMES, start=1)
        ]
        self.jd_profile: dict[str, Any] | None = None
        self.weights: dict[str, float] | None = None
        self.ranked: list[dict[str, Any]] | None = None
        self.validation: dict[str, Any] | None = None
        self._t_start: float | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start_new_run(self) -> str:
        """Reset state for a fresh run. Returns the new run_id."""
        with self._lock:
            self._reset_locked()
            self.status = "queued"
            self.started_at = datetime.now(timezone.utc).isoformat()
            self._t_start = time.perf_counter()
            return self.run_id

    def mark_running(self) -> None:
        with self._lock:
            self.status = "running"

    def step(self, index: int) -> _Step:
        """Get a step by 1-based index (for .start() / .finish() / .fail())."""
        with self._lock:
            return self.steps[index - 1]

    def set_jd_profile(self, jd_profile: dict[str, Any]) -> None:
        with self._lock:
            # Drop raw_text — it's large and not needed by the frontend
            self.jd_profile = {k: v for k, v in jd_profile.items() if k != "raw_text"}

    def set_weights(self, weights: dict[str, float]) -> None:
        with self._lock:
            self.weights = dict(weights)

    def complete(self, ranked: list[dict[str, Any]], validation: dict[str, Any]) -> None:
        with self._lock:
            self.ranked = ranked
            self.validation = validation
            self.status = "completed"
            self.finished_at = datetime.now(timezone.utc).isoformat()

    def fail(self, error_message: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = error_message
            self.finished_at = datetime.now(timezone.utc).isoformat()

    def elapsed_seconds(self) -> float | None:
        with self._lock:
            if self._t_start is None:
                return None
            return round(time.perf_counter() - self._t_start, 2)

    def to_status_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id"          : self.run_id,
                "status"          : self.status,
                "started_at"      : self.started_at,
                "finished_at"     : self.finished_at,
                "elapsed_seconds" : self.elapsed_seconds(),
                "steps"           : [s.as_dict() for s in self.steps],
                "error"           : self.error,
                "result_count"    : len(self.ranked) if self.ranked else None,
            }

    def is_busy(self) -> bool:
        with self._lock:
            return self.status in ("queued", "running")


# Module-level singleton — see Design note above.
run_state = RunState()
