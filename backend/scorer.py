"""
scorer.py — Per-candidate scoring layer for the candidate ranking system.

Each function is pure (no I/O, no global state) so it can be unit-tested in
isolation and called in any order.

Score contract
--------------
Every sub-score is a float in [0.0, 1.0].
`hybrid_score` returns a weighted sum, also in [0.0, 1.0] when weights sum to 1.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

try:
    from config import SENIORITY_YOE, WEIGHTS
except ImportError:
    WEIGHTS = {
        "semantic"      : 0.30,
        "skill_match"   : 0.20,
        "experience"    : 0.15,
        "career_history": 0.10,
        "behaviour"     : 0.10,
        "education"     : 0.05,
        "location"      : 0.10,
    }
    SENIORITY_YOE = {
        "junior": (0,  3),
        "mid"   : (3,  6),
        "senior": (6, 12),
        "lead"  : (10, 99),
    }

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Career-progression constants
# ─────────────────────────────────────────────────────────────────────────────

# Each title keyword maps to a seniority tier (higher = more senior).
# Tiers are intentionally coarse so noisy titles still produce useful signals.
_TITLE_TIER: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"intern|trainee|apprentice",              re.I), 0),
    (re.compile(r"junior|jr\.?\b|associate|entry",        re.I), 1),
    (re.compile(r"\bmid\b|intermediate",                  re.I), 2),
    (re.compile(r"senior|sr\.?\b|staff|principal|expert", re.I), 3),
    (re.compile(r"\blead\b|tech lead|architect",          re.I), 4),
    (re.compile(r"manager|head of|director|vp\b|vice president|cto|ceo|founder", re.I), 5),
]

# Well-known "tier-1" / "tier-2" tech companies for company quality signal
_TIER1_COMPANIES = frozenset({
    "google", "meta", "facebook", "apple", "amazon", "microsoft", "netflix",
    "openai", "anthropic", "deepmind", "nvidia", "salesforce", "adobe",
    "stripe", "databricks", "snowflake", "uber", "airbnb", "linkedin",
    "twitter", "x", "bytedance", "tencent", "alibaba", "baidu",
    "flipkart", "swiggy", "zomato", "razorpay", "phonepe", "paytm",
    "infosys", "wipro", "tcs", "hcl", "thoughtworks", "atlassian",
})
_TIER2_COMPANIES = frozenset({
    "jpmorgan", "goldman sachs", "morgan stanley", "deloitte", "mckinsey",
    "accenture", "ibm", "oracle", "sap", "vmware", "cisco", "qualcomm",
    "samsung", "lg", "sony", "intel", "amd", "arm",
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float, steepness: float = 1.0) -> float:
    """Standard sigmoid: maps any real to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-steepness * x))


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _title_tier(title: str) -> int:
    """Return a seniority tier integer [0, 5] for a job title string."""
    for pattern, tier in _TITLE_TIER:
        if pattern.search(title):
            return tier
    return 2   # default: mid-level when ambiguous


def _company_quality(company: str) -> float:
    """Return 1.0 / 0.75 / 0.5 based on rough company tier."""
    name = company.lower().strip()
    if any(t in name for t in _TIER1_COMPANIES):
        return 1.0
    if any(t in name for t in _TIER2_COMPANIES):
        return 0.75
    return 0.50   # unknown / startup — neutral, not penalising


# ─────────────────────────────────────────────────────────────────────────────
# Individual scorers
# ─────────────────────────────────────────────────────────────────────────────

def score_experience(
    candidate_features: dict[str, Any],
    jd_profile: dict[str, Any],
) -> float:
    """
    Sigmoid-normalised experience fit score.

    Logic
    -----
    1. Extract JD min/max from jd_profile["experience"] (ExperienceRange dict).
       Fall back to SENIORITY_YOE[seniority] if not present.
    2. Use candidate's ``relevant_experience_years`` (preferred) or
       ``total_experience_years`` as the raw value.
    3. Score:
       • candidate_exp >= jd_min          → sigmoid around ideal mid-point → [0.5, 1.0]
       • candidate_exp < jd_min           → penalty proportional to shortfall
       • candidate_exp >> jd_max (+3 yrs) → mild over-qualification penalty
    """
    cand_exp = float(
        candidate_features.get("relevant_experience_years") or
        candidate_features.get("total_experience_years") or 0.0
    )

    # Determine JD experience range
    jd_exp = jd_profile.get("experience") or {}
    jd_min = float(jd_exp.get("min_years", 0))
    jd_max = float(jd_exp.get("max_years", 99))

    if jd_min == 0 and jd_max == 99:
        # Fallback: use seniority band
        seniority = (jd_profile.get("seniority") or "mid").lower()
        band = SENIORITY_YOE.get(seniority, (3, 6))
        jd_min, jd_max = float(band[0]), float(band[1])

    ideal = (jd_min + min(jd_max, jd_min + 6)) / 2.0   # midpoint, capped at +6 yrs

    if cand_exp < jd_min:
        # Under-qualified: linearly penalise from jd_min down to 0
        shortfall = jd_min - cand_exp
        # If jd_min == 0 no penalty possible
        penalty = (shortfall / jd_min) if jd_min > 0 else 0.0
        raw = _clamp(0.5 - 0.5 * penalty)   # [0.0, 0.5)
    elif jd_max < 99 and cand_exp > jd_max + 3:
        # Over-qualified: mild penalty, never below 0.5 so it still ranks OK
        excess = cand_exp - (jd_max + 3)
        raw = _clamp(1.0 - 0.05 * excess, lo=0.5)
    else:
        # Within or slightly above range: sigmoid centred on ideal
        deviation = cand_exp - ideal
        raw = _sigmoid(deviation, steepness=0.4)   # steepness < 1 = gentle curve
        raw = _clamp(0.5 + 0.5 * (raw - 0.5) * 2)  # rescale sigmoid [0,1]→[0.5,1.0]

    return round(_clamp(raw), 4)


def score_career_history(
    candidate_features: dict[str, Any],
    raw_candidate: dict[str, Any] | None = None,
) -> float:
    """
    Career progression score based on role-tier trajectory and company quality.

    Signals (averaged with equal weight if available):
      1. Progression score  — does the candidate move UP the tier ladder?
      2. Company quality    — average tier score across employers
      3. Tenure consistency — not too many short stints (< 12 months)

    Falls back gracefully when raw_candidate is None (only features available).
    """
    # ── Signal 1: progression ─────────────────────────────────────────────────
    progression_score = 0.5   # neutral default

    experiences: list[dict[str, Any]] = []
    if raw_candidate is not None:
        experiences = list(
            raw_candidate.get("career_history") or raw_candidate.get("work_experience") or []
        )

    if len(experiences) >= 2:
        # Sort by start date (best-effort string sort works for ISO dates)
        def _sort_key(e: dict[str, Any]) -> str:
            return str(e.get("start_date") or e.get("from") or "0000")

        try:
            experiences.sort(key=_sort_key)
        except Exception:
            pass

        tiers = [
            _title_tier(str(e.get("title") or e.get("role") or ""))
            for e in experiences
        ]
        # Count upward moves vs downward moves
        up = sum(1 for a, b in zip(tiers, tiers[1:]) if b > a)
        dn = sum(1 for a, b in zip(tiers, tiers[1:]) if b < a)
        total_moves = len(tiers) - 1
        if total_moves > 0:
            # Net upward fraction → [0, 1]
            progression_score = _clamp((up - dn * 0.5) / total_moves)

    # ── Signal 2: company quality ─────────────────────────────────────────────
    company_score = 0.5
    if experiences:
        qualities = [
            _company_quality(str(e.get("company") or e.get("employer") or ""))
            for e in experiences
        ]
        company_score = sum(qualities) / len(qualities)

    # ── Signal 3: tenure consistency ──────────────────────────────────────────
    tenure_score = 0.75   # default: assume OK
    if experiences:
        short_stints = 0
        for e in experiences:
            start = str(e.get("start_date") or e.get("from") or "")
            end   = str(e.get("end_date")   or e.get("to")   or "")
            if not start:
                continue
            # Simple year extraction for a fast, dependency-free duration check
            try:
                sy = int(start[:4])
                ey = int(end[:4]) if end and end[:4].isdigit() else 9999
                months = max(0, (ey - sy) * 12)
                if 0 < months < 12:
                    short_stints += 1
            except (ValueError, IndexError):
                pass
        ratio = short_stints / max(len(experiences), 1)
        tenure_score = _clamp(1.0 - ratio)   # more short stints → lower score

    signals = [progression_score, company_score, tenure_score]
    raw = sum(signals) / len(signals)
    return round(_clamp(raw), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Master score assembler
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_scores(
    candidate_features: dict[str, Any],
    jd_profile: dict[str, Any],
    semantic_score: float,
    raw_candidate: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Assemble all sub-scores for one candidate into a single flat dict.

    Parameters
    ----------
    candidate_features : flat feature dict from ``extract_features()``
    jd_profile         : JDProfile dict from ``parse_jd()``
    semantic_score     : cosine similarity from FAISS (float in [-1, 1])
    raw_candidate      : optional original JSONL dict for career history detail

    Returns
    -------
    Dict mapping score-name → float in [0.0, 1.0]:
      semantic_similarity, skill_match, experience_score,
      career_history, behaviour_score, education_score,
      location_score, quality_score
    """
    # semantic: FAISS cosine scores are in [-1, 1]; rescale to [0, 1]
    sem = _clamp((float(semantic_score) + 1.0) / 2.0)

    # skill_match and education/behaviour/location/quality come pre-computed
    skill_match   = _clamp(float(candidate_features.get("skill_match_score",    0.0)))
    behaviour     = _clamp(float(candidate_features.get("behaviour_score",      0.5)))
    education     = _clamp(float(candidate_features.get("education_score",      0.5)))
    location      = _clamp(float(candidate_features.get("location_match",       0.0)))
    quality       = _clamp(float(candidate_features.get("profile_quality_score",1.0)))

    # experience: computed here with JD context
    try:
        experience = score_experience(candidate_features, jd_profile)
    except Exception as exc:
        logger.warning("score_experience failed: %s", exc)
        experience = 0.5

    # career history: needs raw candidate when available
    try:
        career_history = score_career_history(candidate_features, raw_candidate)
    except Exception as exc:
        logger.warning("score_career_history failed: %s", exc)
        career_history = 0.5

    return {
        "semantic_similarity": round(sem,           4),
        "skill_match"        : round(skill_match,   4),
        "experience_score"   : round(experience,    4),
        "career_history"     : round(career_history,4),
        "behaviour_score"    : round(behaviour,     4),
        "education_score"    : round(education,     4),
        "location_score"     : round(location,      4),
        "quality_score"      : round(quality,       4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid weighted sum
# ─────────────────────────────────────────────────────────────────────────────

# Maps score-dict keys → WEIGHTS keys
_SCORE_TO_WEIGHT_KEY: dict[str, str] = {
    "semantic_similarity": "semantic",
    "skill_match"        : "skill_match",
    "experience_score"   : "experience",
    "career_history"     : "career_history",
    "behaviour_score"    : "behaviour",
    "education_score"    : "education",
    "location_score"     : "location",
    # quality_score is used as a multiplicative post-processor, not a additive term
}


def hybrid_score(
    scores_dict: dict[str, float],
    weights_dict: dict[str, float] | None = None,
    quality_dampen: bool = True,
) -> float:
    """
    Compute the final hybrid ranking score as a weighted sum of sub-scores,
    optionally dampened by the profile-quality signal.

    Parameters
    ----------
    scores_dict    : output of ``compute_all_scores``
    weights_dict   : weight mapping (defaults to config.WEIGHTS).
                     Keys use the config naming convention (e.g. "semantic",
                     "skill_match") and must sum to 1.0.
    quality_dampen : if True, multiply the weighted sum by quality_score so
                     low-quality profiles can't rank in the top-100 regardless
                     of semantic similarity.

    Returns
    -------
    float in [0.0, 1.0]
    """
    w = weights_dict if weights_dict is not None else WEIGHTS

    total = 0.0
    weight_used = 0.0

    for score_key, weight_key in _SCORE_TO_WEIGHT_KEY.items():
        score_val  = float(scores_dict.get(score_key, 0.0))
        weight_val = float(w.get(weight_key, 0.0))
        total      += score_val * weight_val
        weight_used += weight_val

    # Normalise in case weights don't cover all keys (defensive)
    if weight_used > 0 and abs(weight_used - 1.0) > 1e-6:
        total /= weight_used

    # quality_score acts as a multiplier: 1.0 → no change; 0.5 → halved
    if quality_dampen:
        q = float(scores_dict.get("quality_score", 1.0))
        # Soft dampen: score × (0.7 + 0.3×q) — never zeroes a great candidate
        # for a minor quality issue, but a severely penalised profile (q=0) → ×0.7
        total *= (0.7 + 0.3 * q)

    return round(_clamp(total), 6)
