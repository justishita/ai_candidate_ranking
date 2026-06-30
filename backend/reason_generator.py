"""
reason_generator.py — Template-based reasoning string generator.

No LLM calls. Produces a recruiter-friendly, factual sentence under 100 words
for each ranked candidate by combining their scores and features.

Template structure
------------------
  [Fit label]: [X yrs] of [role] experience with expertise in [top skills].
  [Behaviour clause if good.] [Location clause if matched.]
  [Education clause if notable.] Score: X.XXXX

Fit labels
----------
  score >= 0.80  →  "Exceptional fit"
  score >= 0.65  →  "Strong fit"
  score >= 0.50  →  "Good fit"
  score >= 0.35  →  "Moderate fit"
  else           →  "Potential fit"
"""

from __future__ import annotations

import re
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_FIT_LABELS: list[tuple[float, str]] = [
    (0.80, "Exceptional fit"),
    (0.65, "Strong fit"),
    (0.50, "Good fit"),
    (0.35, "Moderate fit"),
    (0.00, "Potential fit"),
]

# How many matching skills to surface in the reason string
_MAX_SKILLS_SHOWN = 4

# Score thresholds for conditional clauses
_BEHAVIOUR_THRESHOLD  = 0.70   # "Highly responsive and active on platform"
_LOCATION_THRESHOLD   = 0.75   # "Based in preferred location"
_EDUCATION_THRESHOLD  = 0.85   # "Holds an advanced degree"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_label(score: float) -> str:
    for threshold, label in _FIT_LABELS:
        if score >= threshold:
            return label
    return "Potential fit"


def _top_matching_skills(
    candidate_features: dict[str, Any],
    jd_mandatory_skills: list[str],
    max_skills: int = _MAX_SKILLS_SHOWN,
) -> list[str]:
    """
    Return the candidate's skills that appear in the JD mandatory list,
    capped at max_skills and formatted for display (title-cased where sensible).
    Falls back to the first N candidate skills if no JD skills are provided.
    """
    raw_skills: list[str] = []

    # skills_list may be a pipe-separated string (DataFrame path) or a list
    sl = candidate_features.get("skills_list", [])
    if isinstance(sl, str):
        raw_skills = [s.strip() for s in sl.split("|") if s.strip()]
    elif isinstance(sl, list):
        raw_skills = [str(s).strip() for s in sl if s]

    if not raw_skills:
        return []

    mandatory_lower = {s.lower() for s in (jd_mandatory_skills or [])}

    # Preferred: intersection with JD mandatory skills
    matched = [s for s in raw_skills if s.lower() in mandatory_lower]

    # Fallback: any candidate skills if no intersection
    display = matched if matched else raw_skills

    # Pretty-print: uppercase known acronyms, else keep as-is
    _CAPS = {
        "aws", "gcp", "sql", "api", "ml", "ai", "ci/cd", "tdd",
        "rest", "grpc", "s3", "gcs", "gpu", "llm", "nlp",
    }
    formatted = []
    for s in display[:max_skills]:
        formatted.append(s.upper() if s.lower() in _CAPS else s)

    return formatted


def _experience_phrase(candidate_features: dict[str, Any], jd_profile: dict[str, Any]) -> str:
    """
    Build "X years of <role> experience" phrase.
    Prefers relevant_experience_years; falls back to total.
    """
    rel_exp   = float(candidate_features.get("relevant_experience_years") or 0.0)
    total_exp = float(candidate_features.get("total_experience_years")    or 0.0)
    yrs       = rel_exp if rel_exp > 0 else total_exp

    role = (jd_profile.get("role_title") or "engineering").strip()
    # Strip seniority prefix from role for the phrase (avoid "Senior Senior …")
    role = re.sub(
        r"^(junior|mid|senior|sr\.?|lead|principal|staff)\s+",
        "", role, flags=re.IGNORECASE
    ).strip()

    if yrs == 0:
        return f"experience in {role}"

    yrs_str = f"{yrs:.0f}" if yrs == int(yrs) else f"{yrs:.1f}"
    return f"{yrs_str} years of {role} experience"


def _education_clause(education_score: float) -> str:
    if education_score >= 1.00:
        return "Holds a PhD or doctoral qualification."
    if education_score >= _EDUCATION_THRESHOLD:
        return "Holds an advanced degree (Master's or equivalent)."
    return ""


def _truncate_to_word_limit(text: str, max_words: int = 100) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    # Find last sentence boundary before the limit
    truncated = " ".join(words[:max_words])
    last_period = truncated.rfind(".")
    if last_period > len(truncated) // 2:
        return truncated[: last_period + 1].strip()
    return truncated.rstrip(",;") + "."


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_reason(
    candidate: dict[str, Any],
    jd_profile: dict[str, Any],
    scores: dict[str, float],
) -> str:
    """
    Generate a recruiter-friendly reasoning string for one ranked candidate.

    Parameters
    ----------
    candidate   : either the raw JSONL dict OR the feature dict from
                  extract_features() — the function checks both field names.
    jd_profile  : JDProfile dict from parse_jd()
    scores      : sub-score dict from compute_all_scores(), plus a top-level
                  "score" key (the final hybrid/LightGBM score).

    Returns
    -------
    str  ≤ 100 words, no newlines, recruiter-readable.
    """
    final_score       = float(scores.get("score", scores.get("hybrid_score", 0.0)))
    behaviour_score   = float(scores.get("behaviour_score", 0.0))
    location_score    = float(scores.get("location_score",  0.0))
    education_score   = float(scores.get("education_score", 0.5))
    skill_match_score = float(scores.get("skill_match",     0.0))

    # ── Fit label ─────────────────────────────────────────────────────────────
    label = _fit_label(final_score)

    # ── Experience phrase ─────────────────────────────────────────────────────
    exp_phrase = _experience_phrase(candidate, jd_profile)

    # ── Top matching skills ───────────────────────────────────────────────────
    top_skills = _top_matching_skills(
        candidate,
        jd_profile.get("mandatory_skills", []),
    )

    # ── Core sentence ─────────────────────────────────────────────────────────
    if top_skills:
        skills_str  = ", ".join(top_skills[:-1])
        if len(top_skills) > 1:
            skills_str += f" and {top_skills[-1]}"
        else:
            skills_str = top_skills[0]
        core = f"{label}: {exp_phrase} with expertise in {skills_str}."
    else:
        core = f"{label}: {exp_phrase}."

    # ── Skill coverage addendum ───────────────────────────────────────────────
    clauses: list[str] = []

    mandatory = jd_profile.get("mandatory_skills") or []
    if mandatory and skill_match_score >= 0.75:
        n_matched = round(skill_match_score * len(mandatory))
        clauses.append(
            f"Covers {n_matched}/{len(mandatory)} required skills."
        )

    # ── Behaviour clause ──────────────────────────────────────────────────────
    if behaviour_score >= _BEHAVIOUR_THRESHOLD:
        if behaviour_score >= 0.90:
            clauses.append("Highly responsive, recently active, and strong platform signals.")
        else:
            clauses.append("Responsive and active on platform.")

    # ── Location clause ───────────────────────────────────────────────────────
    if location_score >= _LOCATION_THRESHOLD:
        locations = jd_profile.get("locations") or []
        if location_score == 1.0 and locations:
            clauses.append(f"Based in {locations[0]}.")
        elif location_score >= 0.80:
            clauses.append("Open to remote; aligns with role flexibility.")
        else:
            clauses.append("Location aligns with preferred region.")

    # ── Education clause ──────────────────────────────────────────────────────
    edu_clause = _education_clause(education_score)
    if edu_clause:
        clauses.append(edu_clause)

    # ── Score tag ─────────────────────────────────────────────────────────────
    clauses.append(f"Score: {final_score:.4f}")

    # ── Assemble ──────────────────────────────────────────────────────────────
    parts = [core] + clauses
    full  = " ".join(parts)
    return _truncate_to_word_limit(full, max_words=100)


def generate_reasons_bulk(
    ranked_candidates: list[dict[str, Any]],
    jd_profile: dict[str, Any],
    raw_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """
    Generate reasoning strings for a list of ranked candidate dicts in one call.

    Parameters
    ----------
    ranked_candidates : list output of ranker.rank_candidates
    jd_profile        : JDProfile from jd_parser.parse_jd
    raw_lookup        : optional {candidate_id → raw JSONL dict} for richer
                        skill resolution; falls back to ranked dict fields

    Returns
    -------
    List of reason strings, same order as ranked_candidates.
    """
    reasons: list[str] = []
    for r in ranked_candidates:
        cid      = str(r.get("candidate_id", ""))
        cand_src = (raw_lookup or {}).get(cid, r)   # prefer raw for skills
        try:
            reason = generate_reason(cand_src, jd_profile, r)
        except Exception:
            reason = f"Score: {r.get('score', 0.0):.4f}"
        reasons.append(reason)
    return reasons
