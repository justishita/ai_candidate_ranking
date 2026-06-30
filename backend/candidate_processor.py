"""
candidate_processor.py — Feature extraction for every candidate in candidates.jsonl.

Design principles
-----------------
• Memory-efficient: candidates.jsonl is streamed line-by-line via a generator;
  the full 100 k profiles are never loaded into RAM at once.
• Pure Python + pandas/numpy — no spaCy, no API calls.
• All scoring functions return a float in [0.0, 1.0].
• `build_text_for_embedding` produces a clean, token-bounded string ready for
  BGE sentence-transformer inference.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator, Iterator

import pandas as pd

# orjson parses JSON lines ~3-5x faster than the stdlib json module, which
# matters a lot when streaming a 100k+ line / 400+ MB candidates.jsonl file.
# Fall back gracefully if it isn't installed so this module still works
# out of the box.
try:
    import orjson
    _loads = orjson.loads
except ImportError:  # pragma: no cover - exercised only when orjson missing
    _loads = json.loads

try:
    from config import CANDIDATES_PATH, TECH_TAXONOMY
except ImportError:
    CANDIDATES_PATH = Path("data/candidates.jsonl")
    TECH_TAXONOMY: list[str] = []

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Education degree → score mapping (checked in priority order)
_EDU_SCORE: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"ph\.?d|doctorate|doctoral",        re.I), 1.00),
    (re.compile(r"m\.?s\.?|master|mtech|m\.?e\b",   re.I), 0.85),
    (re.compile(r"b\.?tech|b\.?e\b|bachelor|b\.?s", re.I), 0.70),
]
_EDU_DEFAULT = 0.50

# Keyword-stuffing threshold
_SKILL_STUFFING_THRESHOLD = 40

# Inactivity thresholds (days)
_INACTIVE_SOFT = 90    # mild penalty starts here
_INACTIVE_HARD = 180   # full penalty cap

# Career gap threshold (months)
_GAP_THRESHOLD_MONTHS = 12

# Embedding text: rough word-to-token ratio for BGE tokenizer (≈ 1.3 tokens/word)
_WORDS_PER_TOKEN = 0.75           # conservative: keep well under 512 tokens
_MAX_WORDS       = int(512 * _WORDS_PER_TOKEN)   # ≈ 384 words

# ─────────────────────────────────────────────────────────────────────────────
# JSONL streaming
# ─────────────────────────────────────────────────────────────────────────────

def stream_candidates(
    path: Path | str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Yield one parsed candidate dict per line from a .jsonl file.

    Skips blank lines and logs (but does not raise on) JSON parse errors so a
    single malformed record never aborts a 100 k-row run.
    """
    jsonl_path = Path(path) if path is not None else CANDIDATES_PATH

    try:
        fh = open(jsonl_path, "r", encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise FileNotFoundError(f"candidates.jsonl not found at: {jsonl_path}")

    with fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield _loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Skipping line %d — JSON error: %s", lineno, exc)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory candidate cache (process-lifetime, keyed by path + mtime)
# ─────────────────────────────────────────────────────────────────────────────
# Step 5 of the pipeline used to re-stream and re-parse the entire JSONL on
# every single run just to pull features for the FAISS-retrieved subset. That
# full re-read/re-parse was the dominant cost on repeat runs (the embedding
# step is already cached to disk by embedder.py). Caching the parsed records
# in memory means only the *first* run in a server process pays the JSON
# parsing cost — every run after that is a dict lookup, which is what makes
# subsequent runs complete in a few seconds instead of re-scanning the file.
_candidate_cache: dict[str, dict[str, Any]] = {}
_candidate_cache_key: tuple[str, float] | None = None


def load_candidate_lookup_cached(
    path: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Return {candidate_id: raw_candidate_dict} for the whole JSONL file,
    parsing it only once per process per (path, mtime). Subsequent calls
    with the same file return the cached dict immediately.
    """
    global _candidate_cache, _candidate_cache_key

    jsonl_path = Path(path) if path is not None else CANDIDATES_PATH
    mtime = jsonl_path.stat().st_mtime
    key = (str(jsonl_path), mtime)

    if key == _candidate_cache_key and _candidate_cache:
        return _candidate_cache

    lookup: dict[str, dict[str, Any]] = {}
    for cand in stream_candidates(jsonl_path):
        cid = str(cand.get("candidate_id") or cand.get("id") or cand.get("_id") or "")
        if cid:
            lookup[cid] = cand

    _candidate_cache = lookup
    _candidate_cache_key = key
    logger.info("Candidate cache (re)built: %d records from %s", len(lookup), jsonl_path)
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Unified single-pass loader (raw lookup + embedding texts together)
# ─────────────────────────────────────────────────────────────────────────────
# Previously the JSONL file was streamed and parsed TWICE on a cold run: once
# in embedder.build_candidate_embeddings() to build embedding texts, and once
# in load_candidate_lookup_cached() to build the raw feature-extraction
# lookup. On a 100k+ row / 400+ MB file, that second full parse is pure
# wasted work — build_text_for_embedding() also gets recomputed redundantly
# in extract_features() for every candidate in the FAISS pool. This function
# streams the file ONCE, builds the raw lookup dict AND the embedding text
# list/ids together, and caches all three in memory keyed by (path, mtime) so
# every later call (embedding step, feature step, repeat runs) is free.
_text_cache: dict[str, str] = {}
_ids_cache: list[str] = []
_texts_cache_key: tuple[str, float] | None = None


def load_candidates_unified_cached(
    path: Path | str | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """
    Single-pass loader: stream candidates.jsonl exactly once and return
    (raw_lookup, candidate_ids, embedding_texts) — all three in the same
    insertion order, with results cached in memory per (path, mtime).

    Replaces doing this work twice (once for embeddings, once for raw
    feature lookup), which used to dominate cold-start runtime at 100k+
    candidates.
    """
    global _candidate_cache, _candidate_cache_key, _text_cache, _ids_cache, _texts_cache_key

    jsonl_path = Path(path) if path is not None else CANDIDATES_PATH
    mtime = jsonl_path.stat().st_mtime
    key = (str(jsonl_path), mtime)

    if key == _candidate_cache_key and key == _texts_cache_key and _candidate_cache:
        return _candidate_cache, list(_ids_cache), [_text_cache[cid] for cid in _ids_cache]

    lookup: dict[str, dict[str, Any]] = {}
    ids: list[str] = []
    texts: list[str] = []
    text_by_id: dict[str, str] = {}

    for cand in stream_candidates(jsonl_path):
        cid = str(cand.get("candidate_id") or cand.get("id") or cand.get("_id") or "")
        if not cid:
            continue
        lookup[cid] = cand
        text = build_text_for_embedding(cand)
        ids.append(cid)
        texts.append(text)
        text_by_id[cid] = text

    _candidate_cache = lookup
    _candidate_cache_key = key
    _text_cache = text_by_id
    _ids_cache = ids
    _texts_cache_key = key

    logger.info(
        "Unified candidate cache (re)built: %d records (single JSONL pass) from %s",
        len(lookup), jsonl_path,
    )
    return lookup, ids, texts


# ─────────────────────────────────────────────────────────────────────────────
# Date / duration utilities
# ─────────────────────────────────────────────────────────────────────────────

_DATE_FMTS = ["%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%m/%Y", "%b %Y", "%B %Y", "%Y"]

def _parse_date(val: Any) -> date | None:
    """Try several common date formats; return None on failure."""
    if val is None:
        return None
    s = str(val).strip()
    if re.match(r"present|current|now", s, re.I):
        return date.today()
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _duration_years(start: Any, end: Any) -> float:
    """Return duration in fractional years; 0.0 on parse failure."""
    s = _parse_date(start)
    e = _parse_date(end) or date.today()
    if s is None or e < s:
        return 0.0
    return (e - s).days / 365.25


def _gap_months(end_prev: Any, start_next: Any) -> float:
    """Return gap between two sequential roles in months (0 if overlapping)."""
    ep = _parse_date(end_prev)
    sn = _parse_date(start_next)
    if ep is None or sn is None or sn <= ep:
        return 0.0
    return (sn - ep).days / 30.44


# ─────────────────────────────────────────────────────────────────────────────
# Skill normalisation
# ─────────────────────────────────────────────────────────────────────────────

_NON_ALNUM = re.compile(r"[^a-z0-9#+.\-/]")

def _normalise_skill(s: str) -> str:
    """Lower-case and collapse whitespace; preserve meaningful punctuation."""
    return _NON_ALNUM.sub(" ", s.lower()).strip()


def _extract_skills_list(candidate: dict[str, Any]) -> list[str]:
    """
    Collect skills from multiple candidate fields:
    skills, technical_skills, tools, keywords, certifications.
    Return a deduplicated, normalised list.
    """
    raw: list[str] = []
    for field in ("skills", "technical_skills", "tools", "keywords", "technologies"):
        val = candidate.get(field)
        if isinstance(val, list):
            raw.extend(str(v) for v in val)
        elif isinstance(val, str):
            # Comma- or pipe-separated strings
            raw.extend(re.split(r"[,|;]", val))

    # Also scan summary / about for taxonomy hits
    for field in ("summary", "about", "objective"):
        blob = candidate.get(field, "") or ""
        for kw in TECH_TAXONOMY:
            if re.search(rf"\b{re.escape(kw)}\b", blob, re.I):
                raw.append(kw)

    seen: set[str] = set()
    out: list[str] = []
    for s in raw:
        norm = _normalise_skill(s)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractors  (each returns a Python scalar)
# ─────────────────────────────────────────────────────────────────────────────

def _total_experience_years(candidate: dict[str, Any]) -> float:
    """
    Sum durations of all work experience entries.
    Falls back to an explicit `total_experience` / `years_of_experience` field.
    """
    # Prefer pre-computed field when available
    for field in ("total_experience", "years_of_experience", "experience_years"):
        val = candidate.get(field)
        if val is not None:
            try:
                return max(0.0, float(val))
            except (TypeError, ValueError):
                pass

    total = 0.0
    for exp in candidate.get("work_experience", []) or []:
        total += _duration_years(
            exp.get("start_date") or exp.get("from"),
            exp.get("end_date")   or exp.get("to"),
        )
    return round(total, 2)


def _relevant_experience_years(
    candidate: dict[str, Any],
    jd_mandatory_skills: set[str],
) -> float:
    """
    Sum durations only for roles whose title or description overlaps with JD
    mandatory skills. Returns 0.0 if jd_mandatory_skills is empty.
    """
    if not jd_mandatory_skills:
        return _total_experience_years(candidate)

    total = 0.0
    for exp in candidate.get("work_experience", []) or []:
        title = (exp.get("title") or exp.get("role") or "").lower()
        desc  = (exp.get("description") or exp.get("responsibilities") or "").lower()
        blob  = title + " " + desc
        if any(re.search(rf"\b{re.escape(sk)}\b", blob) for sk in jd_mandatory_skills):
            total += _duration_years(
                exp.get("start_date") or exp.get("from"),
                exp.get("end_date")   or exp.get("to"),
            )
    return round(total, 2)


def _skill_match_score(
    candidate_skills: list[str],
    jd_mandatory_skills: set[str],
) -> float:
    """
    |candidate_skills ∩ jd_mandatory_skills| / |jd_mandatory_skills|.
    Returns 0.0 when mandatory_skills is empty (avoids ZeroDivisionError).
    """
    if not jd_mandatory_skills:
        return 0.0
    candidate_set = set(candidate_skills)
    hits = len(candidate_set & jd_mandatory_skills)
    return round(hits / len(jd_mandatory_skills), 4)


def _education_score(candidate: dict[str, Any]) -> float:
    """
    Scan all education entries for the highest degree match.
    """
    best = _EDU_DEFAULT
    for edu in candidate.get("education", []) or []:
        degree = (
            edu.get("degree") or edu.get("qualification") or
            edu.get("field_of_study") or ""
        )
        for pattern, score in _EDU_SCORE:
            if pattern.search(degree):
                best = max(best, score)
                break
    return best


def _location_match_score(
    candidate: dict[str, Any],
    jd_locations: list[str],
    jd_remote_ok: bool = False,
) -> float:
    """
    Score:
      1.0 — exact city/country match with JD location
      0.8 — candidate open to remote AND JD allows remote
      0.5 — same state / country but different city
      0.0 — no match
    """
    # Remote preference
    remote_field = (
        candidate.get("remote_ok") or
        candidate.get("open_to_remote") or
        candidate.get("work_preference") or ""
    )
    is_remote_open = (
        remote_field is True or
        (isinstance(remote_field, str) and re.search(r"remote", remote_field, re.I))
    )
    if jd_remote_ok and is_remote_open:
        return 0.8

    cand_location = (
        candidate.get("location") or
        candidate.get("city") or
        candidate.get("current_location") or ""
    ).lower().strip()

    if not cand_location or not jd_locations:
        return 0.0

    for jd_loc in jd_locations:
        jd_loc_lower = jd_loc.lower().strip()
        if jd_loc_lower in cand_location or cand_location in jd_loc_lower:
            return 1.0   # exact / substring match

        # Same state/country heuristic: last token (e.g. "India", "CA")
        jd_tokens   = set(re.split(r"[,\s]+", jd_loc_lower))
        cand_tokens = set(re.split(r"[,\s]+", cand_location))
        if jd_tokens & cand_tokens:
            return 0.5

    return 0.0


def _behaviour_score(candidate: dict[str, Any]) -> float:
    """
    Average of 6 normalised behavioural signals (all → [0, 1]).

    Signal            Raw form             Normalisation
    ─────────────────────────────────────────────────────
    open_to_work      bool / str           1 if True else 0
    response_rate     0-100 %              / 100
    last_active_days  int (days ago)       1 − clamp(days/180, 0, 1)
    github_activity   0-100 score          / 100
    interview_completion  0-100 %          / 100
    offer_acceptance      0-100 %          / 100
    """
    signals: list[float] = []

    # 1. open_to_work
    otw = candidate.get("open_to_work") or candidate.get("actively_looking")
    if otw is not None:
        if isinstance(otw, bool):
            signals.append(1.0 if otw else 0.0)
        elif isinstance(otw, str):
            signals.append(1.0 if re.search(r"yes|true|active", otw, re.I) else 0.0)
        else:
            try:
                signals.append(min(1.0, max(0.0, float(otw))))
            except (TypeError, ValueError):
                pass

    # 2. response_rate
    rr = candidate.get("response_rate")
    if rr is not None:
        try:
            signals.append(min(1.0, max(0.0, float(rr) / 100.0)))
        except (TypeError, ValueError):
            pass

    # 3. last_active_days (inverted: fewer days → higher score)
    lad = candidate.get("last_active_days") or candidate.get("days_since_active")
    if lad is not None:
        try:
            days = float(lad)
            signals.append(1.0 - min(1.0, days / 180.0))
        except (TypeError, ValueError):
            pass

    # 4. github_activity
    gh = candidate.get("github_activity") or candidate.get("github_score")
    if gh is not None:
        try:
            signals.append(min(1.0, max(0.0, float(gh) / 100.0)))
        except (TypeError, ValueError):
            pass

    # 5. interview_completion
    ic = candidate.get("interview_completion") or candidate.get("interview_completion_rate")
    if ic is not None:
        try:
            signals.append(min(1.0, max(0.0, float(ic) / 100.0)))
        except (TypeError, ValueError):
            pass

    # 6. offer_acceptance
    oa = candidate.get("offer_acceptance") or candidate.get("offer_acceptance_rate")
    if oa is not None:
        try:
            signals.append(min(1.0, max(0.0, float(oa) / 100.0)))
        except (TypeError, ValueError):
            pass

    return round(sum(signals) / len(signals), 4) if signals else 0.5  # neutral default


def _profile_quality_score(candidate: dict[str, Any], skills_list: list[str]) -> float:
    """
    Start at 1.0 and apply penalty factors:
      −0.3  any career gap > 12 months
      −0.2  keyword-stuffed profile (> 40 skills)
      −0.2  inactive > 180 days
      −0.1  incomplete profile (no summary AND no work experience)
    Score is clamped to [0.0, 1.0].
    """
    score = 1.0

    # ── Penalty: career gaps ─────────────────────────────────────────────────
    experiences = candidate.get("work_experience", []) or []
    if len(experiences) >= 2:
        # Sort by start date
        dated: list[tuple[date | None, date | None]] = []
        for exp in experiences:
            s = _parse_date(exp.get("start_date") or exp.get("from"))
            e = _parse_date(exp.get("end_date")   or exp.get("to"))
            dated.append((s, e or date.today()))
        dated.sort(key=lambda x: x[0] or date.min)

        for i in range(1, len(dated)):
            prev_end   = dated[i - 1][1]
            curr_start = dated[i][0]
            gap = _gap_months(prev_end, curr_start)
            if gap > _GAP_THRESHOLD_MONTHS:
                score -= 0.30
                break   # one penalty is enough

    # ── Penalty: keyword stuffing ────────────────────────────────────────────
    if len(skills_list) > _SKILL_STUFFING_THRESHOLD:
        score -= 0.20

    # ── Penalty: inactivity > 180 days ───────────────────────────────────────
    lad = candidate.get("last_active_days") or candidate.get("days_since_active")
    if lad is not None:
        try:
            if float(lad) > _INACTIVE_HARD:
                score -= 0.20
            elif float(lad) > _INACTIVE_SOFT:
                # proportional penalty between soft and hard thresholds
                frac = (float(lad) - _INACTIVE_SOFT) / (_INACTIVE_HARD - _INACTIVE_SOFT)
                score -= 0.20 * frac
        except (TypeError, ValueError):
            pass

    # ── Penalty: sparse profile ───────────────────────────────────────────────
    has_summary = bool((candidate.get("summary") or candidate.get("about") or "").strip())
    has_work    = bool(experiences)
    if not has_summary and not has_work:
        score -= 0.10

    return round(max(0.0, min(1.0, score)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding text builder
# ─────────────────────────────────────────────────────────────────────────────

def build_text_for_embedding(candidate: dict[str, Any], max_words: int = _MAX_WORDS) -> str:
    """
    Concatenate the richest candidate text fields into a single string suitable
    for BGE / sentence-transformer embedding.  Truncated to ``max_words`` words
    to stay comfortably within the 512-token context window.

    Field priority (higher = included first):
      1. summary / about / objective
      2. top skills (space-separated)
      3. work experience: job title + description (most recent first)
      4. education degrees
    """
    parts: list[str] = []

    # 1. Summary
    for field in ("summary", "about", "objective", "headline"):
        val = (candidate.get(field) or "").strip()
        if val:
            parts.append(val)
            break

    # 2. Skills
    skills = _extract_skills_list(candidate)
    if skills:
        parts.append("Skills: " + ", ".join(skills[:_SKILL_STUFFING_THRESHOLD]))

    # 3. Work experience (most recent first, limited to avoid overflow)
    experiences = list(candidate.get("work_experience", []) or [])
    # Sort descending by start date
    def _sort_key(exp: dict[str, Any]) -> date:
        d = _parse_date(exp.get("start_date") or exp.get("from"))
        return d if d is not None else date.min

    try:
        experiences.sort(key=_sort_key, reverse=True)
    except Exception:
        pass

    for exp in experiences[:6]:      # cap at 6 roles to save token budget
        title = (exp.get("title") or exp.get("role") or "").strip()
        company = (exp.get("company") or exp.get("employer") or "").strip()
        desc  = (exp.get("description") or exp.get("responsibilities") or "").strip()
        # Truncate individual role description
        desc_words = desc.split()
        if len(desc_words) > 60:
            desc = " ".join(desc_words[:60]) + "..."
        role_text = " | ".join(filter(None, [title, company, desc]))
        if role_text:
            parts.append(role_text)

    # 4. Education
    for edu in (candidate.get("education", []) or [])[:3]:
        degree  = (edu.get("degree") or edu.get("qualification") or "").strip()
        school  = (edu.get("institution") or edu.get("school") or edu.get("university") or "").strip()
        edu_str = " ".join(filter(None, [degree, school]))
        if edu_str:
            parts.append(edu_str)

    full_text = " ".join(parts)

    # Token-budget truncation at word level
    words = full_text.split()
    if len(words) > max_words:
        full_text = " ".join(words[:max_words])

    return full_text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main feature-extraction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    candidate: dict[str, Any],
    jd_mandatory_skills: list[str] | set[str] | None = None,
    jd_locations: list[str] | None = None,
    jd_remote_ok: bool = False,
    precomputed_embed_text: str | None = None,
) -> dict[str, Any]:
    """
    Extract all scoring features for a single candidate dict.

    Parameters
    ----------
    candidate              : raw dict parsed from one JSONL line
    jd_mandatory_skills    : list/set of normalised mandatory skill strings from JDProfile
    jd_locations           : list of location strings from JDProfile
    jd_remote_ok           : whether the JD explicitly allows remote
    precomputed_embed_text : pass the embedding text already computed for this
                              candidate (e.g. from load_candidates_unified_cached)
                              to avoid rebuilding it — build_text_for_embedding()
                              is otherwise the same work done twice per candidate.

    Returns
    -------
    Flat dict with all feature columns + the original candidate_id.
    """
    mandatory = set(s.lower().strip() for s in (jd_mandatory_skills or []))
    locations = list(jd_locations or [])

    cid = (
        candidate.get("candidate_id") or
        candidate.get("id") or
        candidate.get("_id") or
        ""
    )

    try:
        skills = _extract_skills_list(candidate)
    except Exception as exc:
        logger.warning("Skill extraction failed for %s: %s", cid, exc)
        skills = []

    try:
        total_exp = _total_experience_years(candidate)
    except Exception as exc:
        logger.warning("total_experience failed for %s: %s", cid, exc)
        total_exp = 0.0

    try:
        rel_exp = _relevant_experience_years(candidate, mandatory)
    except Exception as exc:
        logger.warning("relevant_experience failed for %s: %s", cid, exc)
        rel_exp = 0.0

    try:
        skill_match = _skill_match_score(skills, mandatory)
    except Exception as exc:
        logger.warning("skill_match failed for %s: %s", cid, exc)
        skill_match = 0.0

    try:
        edu_score = _education_score(candidate)
    except Exception as exc:
        logger.warning("education_score failed for %s: %s", cid, exc)
        edu_score = _EDU_DEFAULT

    try:
        loc_score = _location_match_score(candidate, locations, jd_remote_ok)
    except Exception as exc:
        logger.warning("location_match failed for %s: %s", cid, exc)
        loc_score = 0.0

    try:
        beh_score = _behaviour_score(candidate)
    except Exception as exc:
        logger.warning("behaviour_score failed for %s: %s", cid, exc)
        beh_score = 0.5

    try:
        pq_score = _profile_quality_score(candidate, skills)
    except Exception as exc:
        logger.warning("profile_quality failed for %s: %s", cid, exc)
        pq_score = 1.0

    try:
        embed_text = precomputed_embed_text if precomputed_embed_text is not None else build_text_for_embedding(candidate)
    except Exception as exc:
        logger.warning("embed_text failed for %s: %s", cid, exc)
        embed_text = ""

    return {
        "candidate_id"            : cid,
        "total_experience_years"  : total_exp,
        "relevant_experience_years": rel_exp,
        "skills_list"             : skills,          # list — expanded to str for DF
        "skill_match_score"       : skill_match,
        "education_score"         : edu_score,
        "location_match"          : loc_score,
        "behaviour_score"         : beh_score,
        "profile_quality_score"   : pq_score,
        "embedding_text"          : embed_text,
    }


def process_candidates(
    path: Path | str | None = None,
    jd_mandatory_skills: list[str] | set[str] | None = None,
    jd_locations: list[str] | None = None,
    jd_remote_ok: bool = False,
    as_dataframe: bool = True,
) -> pd.DataFrame | list[dict[str, Any]]:
    """
    Stream ``candidates.jsonl``, extract features for each record and return
    the result as a pandas DataFrame (default) or list of dicts.

    Parameters
    ----------
    path                 : override for the JSONL file path
    jd_mandatory_skills  : from JDProfile["mandatory_skills"]
    jd_locations         : from JDProfile["locations"]
    jd_remote_ok         : True if JD mentions "remote"
    as_dataframe         : if True (default), return pd.DataFrame; else list[dict]

    Notes
    -----
    ``skills_list`` is stored as a pipe-separated string in the DataFrame so
    the column stays serialisable to CSV downstream.
    """
    records: list[dict[str, Any]] = []
    total = 0

    for candidate in stream_candidates(path):
        total += 1
        feats = extract_features(
            candidate,
            jd_mandatory_skills=jd_mandatory_skills,
            jd_locations=jd_locations,
            jd_remote_ok=jd_remote_ok,
        )
        records.append(feats)

        if total % 10_000 == 0:
            logger.info("Processed %d candidates …", total)

    logger.info("Feature extraction complete: %d candidates total.", total)

    if not as_dataframe:
        return records

    df = pd.DataFrame(records)

    # Serialise list columns for compatibility with downstream CSV writers
    if "skills_list" in df.columns:
        df["skills_list"] = df["skills_list"].apply(
            lambda s: "|".join(s) if isinstance(s, list) else (s or "")
        )

    # Enforce numeric dtypes on score columns
    score_cols = [
        "total_experience_years", "relevant_experience_years",
        "skill_match_score", "education_score",
        "location_match", "behaviour_score", "profile_quality_score",
    ]
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    ap = argparse.ArgumentParser(description="Extract features from candidates.jsonl")
    ap.add_argument("--jsonl",   type=Path, default=None, help="Path to candidates.jsonl")
    ap.add_argument("--skills",  nargs="*", default=[], help="JD mandatory skills (space-separated)")
    ap.add_argument("--locations", nargs="*", default=[], help="JD locations")
    ap.add_argument("--remote",  action="store_true", help="JD allows remote")
    ap.add_argument("--limit",   type=int, default=5,  help="Print first N rows (default 5)")
    args = ap.parse_args()

    try:
        df = process_candidates(
            path                 = args.jsonl,
            jd_mandatory_skills  = args.skills or None,
            jd_locations         = args.locations or None,
            jd_remote_ok         = args.remote,
            as_dataframe         = True,
        )
    except FileNotFoundError as err:
        sys.exit(f"ERROR: {err}")

    print(f"\nExtracted features for {len(df):,} candidates.")
    print(f"\nColumns: {list(df.columns)}\n")

    display_cols = [c for c in df.columns if c != "embedding_text"]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df[display_cols].head(args.limit).to_string(index=False))

    # Show one embedding text sample
    if not df.empty:
        print("\n── Sample embedding text (first candidate) ──")
        print(df["embedding_text"].iloc[0][:600])
