"""
jd_parser.py — Parse job_description.docx into a structured JDProfile dict.

No external API calls. All extraction is done via:
  • python-docx  — DOCX reading
  • spaCy        — NLP (en_core_web_sm)
  • re            — pattern matching for years / seniority
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import TypedDict
import os


# ── Optional but strongly recommended: rich tracebacks ───────────────────────
try:
    import docx                # python-docx
except ImportError as e:
    sys.exit(f"[jd_parser] python-docx not installed: {e}")

try:
    import spacy
except ImportError as e:
    sys.exit(f"[jd_parser] spaCy not installed: {e}")

# Local import — config lives alongside this file
try:
    from config import JD_PATH, SPACY_MODEL, TECH_TAXONOMY
except ImportError:
    # Fallback so the module is usable stand-alone during development
    JD_PATH       = Path("data/job_description.docx")
    SPACY_MODEL   = "en_core_web_sm"
    TECH_TAXONOMY: list[str] = []

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TypedDict for the returned profile
# ─────────────────────────────────────────────────────────────────────────────

class ExperienceRange(TypedDict):
    min_years: int
    max_years: int   # 99 means "open-ended" (X+ years)
    raw: str


class JDProfile(TypedDict):
    role_title:        str
    seniority:         str                  # junior | mid | senior | lead | unknown
    experience:        ExperienceRange | None
    mandatory_skills:  list[str]
    preferred_skills:  list[str]
    locations:         list[str]
    raw_text:          str                  # full JD text, useful for embedding


# ─────────────────────────────────────────────────────────────────────────────
# Constants / compiled patterns
# ─────────────────────────────────────────────────────────────────────────────

# Match "3-5 years", "3–5 years", "3 to 5 years", "5+ years", "5 or more years"
_YOE_RANGE_RE  = re.compile(
    r"(\d+)\s*(?:-|–|to)\s*(\d+)\s+years?",
    re.IGNORECASE,
)
_YOE_PLUS_RE   = re.compile(
    r"(\d+)\+?\s+(?:or\s+more\s+)?years?",
    re.IGNORECASE,
)

_SENIORITY_MAP: dict[str, str] = {
    "junior"     : "junior",
    "entry.level": "junior",
    "entry level": "junior",
    "associate"  : "junior",
    "mid.level"  : "mid",
    "mid level"  : "mid",
    "intermediate": "mid",
    "senior"     : "senior",
    "sr\."       : "senior",
    "staff"      : "senior",
    "principal"  : "senior",
    "lead"       : "lead",
    "tech lead"  : "lead",
    "engineering lead": "lead",
    "manager"    : "lead",
}
_SENIORITY_RE  = re.compile(
    "|".join(rf"(?P<s{i}>{k})" for i, k in enumerate(_SENIORITY_MAP)),
    re.IGNORECASE,
)

# Section headers that typically introduce preferred / nice-to-have skills
_PREFERRED_HEADERS = re.compile(
    r"(preferred|nice.to.have|bonus|good.to.have|plus|desirable)",
    re.IGNORECASE,
)
# Section headers for mandatory requirements
_REQUIRED_HEADERS = re.compile(
    r"(required|must.have|mandatory|minimum qualif|basic qualif|key requirement)",
    re.IGNORECASE,
)

# Common location indicator phrases
_LOCATION_RE = re.compile(
    r"\b(?:location|based in|office|remote|hybrid|on.?site)[:\s]+([A-Za-z ,/&()-]{3,60})",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_spacy() -> spacy.Language:
    """Load spaCy model; fall back with a clear error."""
    try:
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "lemmatizer"])
        logger.debug("spaCy model '%s' loaded.", SPACY_MODEL)
        return nlp
    except OSError:
        logger.error(
            "spaCy model '%s' not found. Run: python -m spacy download %s",
            SPACY_MODEL, SPACY_MODEL,
        )
        raise


def _read_docx(path: Path) -> str:
    """Extract all paragraph text from a .docx, preserving newlines."""
    print("=" * 50)
    print("Current working directory:", os.getcwd())
    print("Path object:", path)
    print("Absolute path:", path.resolve())
    print("Exists:", path.exists())
    print("=" * 50)
    doc = docx.Document(str(path))
    paragraphs = [para.text.strip() for para in doc.paragraphs]
    return "\n".join(p for p in paragraphs if p)


def _extract_experience(text: str) -> ExperienceRange | None:
    """Return the first well-formed experience mention found in text."""
    # Try range pattern first: "3-5 years"
    m = _YOE_RANGE_RE.search(text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return ExperienceRange(min_years=lo, max_years=hi, raw=m.group(0))

    # Fall back to X+ pattern: "5+ years" or "5 years"
    m = _YOE_PLUS_RE.search(text)
    if m:
        lo = int(m.group(1))
        raw = m.group(0)
        hi  = 99 if "+" in raw or "more" in raw.lower() else lo
        return ExperienceRange(min_years=lo, max_years=hi, raw=raw)

    return None


def _extract_seniority(title: str, text: str) -> str:
    """Detect seniority level from job title first, then full text."""
    for source in (title, text[:500]):          # check title & top of JD
        m = _SENIORITY_RE.search(source)
        if m:
            matched_key = m.group(0).lower().strip()
            for key, level in _SENIORITY_MAP.items():
                if re.fullmatch(key, matched_key, re.IGNORECASE):
                    return level
            # Fallback: iterate named groups
            for key, level in _SENIORITY_MAP.items():
                pattern_fragment = key.replace(".", r"\.")
                if re.search(pattern_fragment, matched_key, re.IGNORECASE):
                    return level
    return "unknown"


def _split_sections(text: str) -> dict[str, str]:
    """
    Naively split JD text into labelled sections by scanning for common
    header lines (all-caps words or title-cased short lines ending in ':').
    Returns a dict of {header_text: section_body}.
    """
    sections: dict[str, str] = {"_preamble": ""}
    current_header = "_preamble"
    lines = text.splitlines()

    for line in lines:
        stripped = line.strip()
        # Heuristic: a header is ≤ 8 words, short, and has no period mid-line
        if (
            stripped
            and len(stripped.split()) <= 8
            and (stripped.isupper() or stripped.endswith(":") or (
                stripped.istitle() and len(stripped) < 60
            ))
        ):
            current_header = stripped.lower().rstrip(":")
            sections.setdefault(current_header, "")
        else:
            sections[current_header] = sections.get(current_header, "") + "\n" + line

    return sections


def _extract_skills(
    text: str,
    nlp: spacy.Language,
    taxonomy: list[str],
) -> tuple[list[str], list[str]]:
    """
    Split text into 'required' vs 'preferred' sections, then match tech
    keywords from the taxonomy using NLP noun-chunks + direct token matching.

    Returns (mandatory_skills, preferred_skills).
    """
    sections = _split_sections(text)

    required_text  : list[str] = []
    preferred_text : list[str] = []

    for header, body in sections.items():
        if _REQUIRED_HEADERS.search(header):
            required_text.append(body)
        elif _PREFERRED_HEADERS.search(header):
            preferred_text.append(body)
        elif header == "_preamble":
            # Treat preamble as required by default
            required_text.append(body)
        else:
            # Unlabelled sections: scan header for hints
            if _PREFERRED_HEADERS.search(header):
                preferred_text.append(body)
            else:
                required_text.append(body)

    taxonomy_lower = [t.lower() for t in taxonomy]

    def _match_skills(blob: str) -> list[str]:
        blob_lower = blob.lower()
        matched = []
        for skill in taxonomy_lower:
            # Word-boundary match to avoid "r" matching "required"
            pattern = rf"\b{re.escape(skill)}\b"
            if re.search(pattern, blob_lower):
                matched.append(skill)

        # Additionally, extract ORG / PRODUCT named entities via spaCy
        try:
            doc = nlp(blob[:50_000])   # spaCy hard-limits on very long strings
            for ent in doc.ents:
                if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
                    candidate = ent.text.lower().strip()
                    if len(candidate) >= 2 and candidate not in matched:
                        matched.append(candidate)
        except Exception:
            pass

        return sorted(set(matched))

    mandatory = _match_skills("\n".join(required_text))
    preferred = _match_skills("\n".join(preferred_text))

    # De-duplicate: don't report a skill as both mandatory and preferred
    preferred = [s for s in preferred if s not in mandatory]

    return mandatory, preferred


def _extract_role_title(text: str, nlp: spacy.Language) -> str:
    """
    Extract job role/title. Strategy:
      1. Look for "Job Title:" or "Position:" prefix in first 20 lines.
      2. Fall back to first noun-chunk or first meaningful line.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    title_prefix_re = re.compile(
        r"(?:job\s+title|position|role|title)\s*[:\-]\s*(.+)",
        re.IGNORECASE,
    )
    for line in lines[:20]:
        m = title_prefix_re.match(line)
        if m:
            return m.group(1).strip()

    # spaCy: first noun chunk in the first 200 chars
    try:
        doc = nlp(text[:200])
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:
                return chunk.text.strip()
    except Exception:
        pass

    # Last resort: first non-empty line
    return lines[0] if lines else "Unknown Role"


def _extract_locations(text: str, nlp: spacy.Language) -> list[str]:
    """
    Collect location mentions via:
      1. Regex for explicit 'Location:' headers.
      2. spaCy GPE + LOC entities.
    """
    locations: list[str] = []

    # Regex hits
    for m in _LOCATION_RE.finditer(text):
        raw = m.group(1).strip().rstrip(".")
        if raw and raw.lower() not in ("the", "a", "an"):
            locations.append(raw)

    # spaCy GPE / LOC entities
    try:
        doc = nlp(text[:30_000])
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC", "FAC"):
                loc = ent.text.strip()
                if loc not in locations:
                    locations.append(loc)
    except Exception:
        pass

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for loc in locations:
        if loc.lower() not in seen:
            seen.add(loc.lower())
            unique.append(loc)

    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_jd(path: Path | str | None = None) -> JDProfile:
    """
    Parse a Job Description .docx file and return a structured JDProfile.

    Parameters
    ----------
    path : Path or str, optional
        Path to the .docx file. Defaults to ``JD_PATH`` from config.

    Returns
    -------
    JDProfile
        Typed dict with all extracted fields.
    """
    jd_path = Path(path) if path is not None else JD_PATH

    # ── Load model once ───────────────────────────────────────────────────────
    try:
        nlp = _load_spacy()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot load spaCy model '{SPACY_MODEL}'. "
            f"Run: python -m spacy download {SPACY_MODEL}"
        ) from exc

    # ── Read DOCX ─────────────────────────────────────────────────────────────
    try:
        raw_text = _read_docx(jd_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"JD file not found: {jd_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to read DOCX '{jd_path}': {exc}") from exc

    if not raw_text.strip():
        logger.warning("JD file '%s' appears to be empty.", jd_path)

    # ── Extract fields ────────────────────────────────────────────────────────
    try:
        role_title = _extract_role_title(raw_text, nlp)
    except Exception as exc:
        logger.warning("Role title extraction failed: %s", exc)
        role_title = "Unknown Role"

    try:
        seniority = _extract_seniority(role_title, raw_text)
    except Exception as exc:
        logger.warning("Seniority extraction failed: %s", exc)
        seniority = "unknown"

    try:
        experience = _extract_experience(raw_text)
    except Exception as exc:
        logger.warning("Experience extraction failed: %s", exc)
        experience = None

    try:
        mandatory_skills, preferred_skills = _extract_skills(
            raw_text, nlp, TECH_TAXONOMY
        )
    except Exception as exc:
        logger.warning("Skill extraction failed: %s", exc)
        mandatory_skills, preferred_skills = [], []

    try:
        locations = _extract_locations(raw_text, nlp)
    except Exception as exc:
        logger.warning("Location extraction failed: %s", exc)
        locations = []

    profile = JDProfile(
        role_title       = role_title,
        seniority        = seniority,
        experience       = experience,
        mandatory_skills = mandatory_skills,
        preferred_skills = preferred_skills,
        locations        = locations,
        raw_text         = raw_text,
    )

    logger.info(
        "JD parsed → role='%s' seniority='%s' mandatory=%d preferred=%d locations=%d",
        role_title, seniority,
        len(mandatory_skills), len(preferred_skills), len(locations),
    )
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Parse a job description .docx file.")
    parser.add_argument(
        "--jd",
        type=Path,
        default=None,
        help="Path to job_description.docx (defaults to config.JD_PATH)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent level for pretty-printing (default: 2)",
    )
    args = parser.parse_args()

    try:
        profile = parse_jd(args.jd)
    except (FileNotFoundError, RuntimeError) as err:
        sys.exit(f"ERROR: {err}")

    # Pretty-print — exclude raw_text from CLI output to keep it readable
    printable = {k: v for k, v in profile.items() if k != "raw_text"}
    printable["raw_text_length"] = len(profile["raw_text"])

    print(json.dumps(printable, indent=args.indent, ensure_ascii=False))
