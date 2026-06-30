"""
output.py — Write and validate the final submission.csv.

Submission schema
-----------------
  candidate_id : str   — unique candidate identifier
  rank         : int   — 1-based rank (1 = best match)
  score        : float — hybrid/LightGBM score, rounded to 4 d.p., in [0, 1]
  reasoning    : str   — recruiter-friendly reason string

Validation rules (validate_output)
-----------------------------------
  1. File exists and is readable.
  2. Exactly 4 columns: candidate_id, rank, score, reasoning.
  3. Exactly 100 rows.
  4. No null / NaN values in any column.
  5. score ∈ [0.0, 1.0] for all rows.
  6. rank is sequential integers 1–100 with no duplicates.
  7. candidate_id values are unique.
  8. reasoning column is non-empty strings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from config import FINAL_TOP_N, SUBMISSION_PATH
except ImportError:
    FINAL_TOP_N     = 100
    SUBMISSION_PATH = Path("output/submission.csv")

logger = logging.getLogger(__name__)

# ── Submission column names (order matters for CSV) ───────────────────────────
_REQUIRED_COLUMNS: list[str] = ["candidate_id", "rank", "score", "reasoning"]


# ─────────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_submission_csv(
    ranked_candidates: list[dict[str, Any]],
    reasons: list[str],
    output_path: Path | str = SUBMISSION_PATH,
    top_n: int = FINAL_TOP_N,
) -> Path:
    """
    Write the ranked candidates and their reasons to a CSV file.

    Parameters
    ----------
    ranked_candidates : list of dicts from ranker.rank_candidates (must have
                        candidate_id, rank, score keys)
    reasons           : list of reason strings — same order / length as
                        ranked_candidates (from reason_generator.generate_reasons_bulk)
    output_path       : destination CSV path (parent dirs are created)
    top_n             : how many rows to write (default: 100)

    Returns
    -------
    Path to the written file.

    Raises
    ------
    ValueError   : if ranked_candidates and reasons have different lengths,
                   or if ranked_candidates is empty.
    """
    if not ranked_candidates:
        raise ValueError("ranked_candidates is empty — nothing to write.")

    if len(ranked_candidates) != len(reasons):
        raise ValueError(
            f"Length mismatch: {len(ranked_candidates)} candidates "
            f"but {len(reasons)} reasons."
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build output rows (top_n only)
    rows: list[dict[str, Any]] = []
    for candidate, reason in zip(ranked_candidates[:top_n], reasons[:top_n]):
        rows.append({
            "candidate_id": str(candidate["candidate_id"]),
            "rank"        : int(candidate["rank"]),
            "score"       : round(float(candidate["score"]), 4),
            "reasoning"   : str(reason).strip(),
        })

    df = pd.DataFrame(rows, columns=_REQUIRED_COLUMNS)

    # ── Sanity coercions before writing ───────────────────────────────────────
    df["rank"]  = df["rank"].astype(int)
    df["score"] = df["score"].clip(0.0, 1.0).round(4)

    # Ensure reasoning is never null
    df["reasoning"] = df["reasoning"].fillna("").str.strip()
    df.loc[df["reasoning"] == "", "reasoning"] = "Score: " + df["score"].astype(str)

    df.to_csv(out_path, index=False)
    logger.info(
        "Submission CSV written → %s  (%d rows)", out_path, len(df)
    )
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when the submission CSV fails one or more validation checks."""


def validate_output(
    path: Path | str = SUBMISSION_PATH,
    expected_rows: int = FINAL_TOP_N,
) -> dict[str, Any]:
    """
    Validate the submission CSV against the hackathon schema requirements.

    Parameters
    ----------
    path          : path to the CSV file to validate
    expected_rows : expected number of data rows (default: 100)

    Returns
    -------
    Dict with validation summary:
      {
        "valid"         : bool,
        "row_count"     : int,
        "checks_passed" : list[str],
        "checks_failed" : list[str],
      }

    Raises
    ------
    ValidationError : if any hard-fail check fails (file missing, wrong columns).
                      Soft checks are collected and returned in "checks_failed".
    """
    csv_path = Path(path)
    checks_passed: list[str] = []
    checks_failed: list[str] = []

    # ── Hard check 1: file exists ─────────────────────────────────────────────
    if not csv_path.exists():
        raise ValidationError(f"Output file not found: {csv_path}")
    checks_passed.append("File exists")

    # ── Load ─────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path, dtype={"candidate_id": str})
    except Exception as exc:
        raise ValidationError(f"Could not read CSV: {exc}") from exc

    # ── Hard check 2: required columns ────────────────────────────────────────
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValidationError(f"Missing columns: {missing_cols}")
    checks_passed.append(f"All required columns present: {_REQUIRED_COLUMNS}")

    # ── Soft check 3: row count ───────────────────────────────────────────────
    n_rows = len(df)
    if n_rows == expected_rows:
        checks_passed.append(f"Row count correct: {n_rows}")
    else:
        checks_failed.append(
            f"Row count: expected {expected_rows}, got {n_rows}"
        )

    # ── Soft check 4: no nulls ────────────────────────────────────────────────
    null_counts = df[_REQUIRED_COLUMNS].isnull().sum()
    if null_counts.sum() == 0:
        checks_passed.append("No null values in any column")
    else:
        for col, cnt in null_counts[null_counts > 0].items():
            checks_failed.append(f"Column '{col}' has {cnt} null(s)")

    # ── Soft check 5: score range [0, 1] ─────────────────────────────────────
    scores = pd.to_numeric(df["score"], errors="coerce")
    invalid_scores = scores.isna() | (scores < 0) | (scores > 1)
    if not invalid_scores.any():
        checks_passed.append("All scores in [0.0, 1.0]")
    else:
        bad = invalid_scores.sum()
        checks_failed.append(f"{bad} score value(s) outside [0, 1] or non-numeric")

    # ── Soft check 6: ranks sequential 1..expected_rows ──────────────────────
    ranks = pd.to_numeric(df["rank"], errors="coerce").dropna().astype(int)
    expected_ranks = set(range(1, n_rows + 1))
    actual_ranks   = set(ranks.tolist())
    if actual_ranks == expected_ranks and not ranks.duplicated().any():
        checks_passed.append(f"Ranks are sequential 1–{n_rows} with no duplicates")
    else:
        missing_r  = sorted(expected_ranks - actual_ranks)
        dup_r      = sorted(ranks[ranks.duplicated()].tolist())
        if missing_r:
            checks_failed.append(f"Missing rank values: {missing_r[:10]}")
        if dup_r:
            checks_failed.append(f"Duplicate rank values: {dup_r[:10]}")

    # ── Soft check 7: unique candidate IDs ───────────────────────────────────
    dup_ids = df["candidate_id"][df["candidate_id"].duplicated()].tolist()
    if not dup_ids:
        checks_passed.append("All candidate_id values are unique")
    else:
        checks_failed.append(f"{len(dup_ids)} duplicate candidate_id(s): {dup_ids[:5]}")

    # ── Soft check 8: non-empty reasoning ────────────────────────────────────
    empty_reasons = (df["reasoning"].isna() | (df["reasoning"].str.strip() == "")).sum()
    if empty_reasons == 0:
        checks_passed.append("All reasoning fields are non-empty")
    else:
        checks_failed.append(f"{empty_reasons} empty reasoning string(s)")

    is_valid = len(checks_failed) == 0

    summary: dict[str, Any] = {
        "valid"         : is_valid,
        "row_count"     : n_rows,
        "checks_passed" : checks_passed,
        "checks_failed" : checks_failed,
    }

    if is_valid:
        logger.info("✓ Submission validated: all %d checks passed.", len(checks_passed))
    else:
        logger.warning(
            "✗ Submission validation: %d passed, %d FAILED → %s",
            len(checks_passed), len(checks_failed), checks_failed,
        )

    return summary
