# AI Candidate Ranking System

Ranks 100,000 candidate profiles against a job description in under 5 minutes on CPU, producing a `submission.csv` with the top 100 candidates, their scores, and recruiter-ready reasoning.

---

## Project Structure

```
project/
├── backend/
│   ├── app.py                  # FastAPI entry point — all API routes
│   ├── pipeline_executor.py    # Runs the pipeline in a background thread,
│   │                           #   reports per-step progress into run_state
│   ├── run_state.py            # Thread-safe in-memory run/progress tracker
│   ├── schemas.py              # Pydantic request/response models
│   ├── main.py                 # CLI entry point (unchanged — still works standalone)
│   ├── config.py                # Paths, model names, weights, runtime constants
│   ├── jd_parser.py             # Parses job_description.docx → structured JDProfile
│   ├── candidate_processor.py   # Streams candidates.jsonl → feature extraction
│   ├── embedder.py              # BGE sentence-transformer encoding + disk cache
│   ├── indexer.py               # FAISS index build, search, save, load
│   ├── scorer.py                # Per-candidate sub-scores + hybrid weighted sum
│   ├── ranker.py                # Re-ranking layer (hybrid + optional LightGBM)
│   ├── reason_generator.py      # Template-based reasoning string generation
│   ├── output.py                # CSV writer + schema validator
│   └── requirements.txt
├── frontend/
│   └── index.html               # Single-file dashboard — fetches live data from the API
├── data/
│   ├── candidates.jsonl
│   ├── job_description.docx
│   ├── candidate_schema.json
│   └── redrob_signals_doc.docx
├── output/
│   ├── submission.csv
│   ├── candidate_index.faiss
│   ├── candidate_embeddings.npy
│   └── candidate_ids_cache.npy
└── models/
    └── ranker.lgb               # Optional LightGBM re-ranker (skipped if absent)
```

`config.py` resolves `data/`, `output/`, and `models/` as siblings of `backend/` (one level up from `PROJECT_ROOT`), so both `app.py` (the API server) and `main.py` (the CLI) read and write the same files regardless of which one triggers a run.

---

## Setup

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the full stack (API + UI)

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** — `app.py` serves `frontend/index.html` directly via FastAPI's static file mount, so no separate frontend server or build step is needed.

In the sidebar, enter the relative paths to your JD and candidates file (or leave blank to use the `config.py` defaults: `data/job_description.docx` and `data/candidates.jsonl`), then click **Run pipeline**. The UI polls `/api/status` every 1.5s and renders live step-by-step progress, then loads the ranked results automatically when the run completes.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness check — confirms ML dependencies import cleanly |
| `POST` | `/api/run` | Trigger a pipeline run (background thread, returns `202`) |
| `GET` | `/api/status` | Poll current run progress (7-step breakdown) |
| `GET` | `/api/jd` | Parsed JD profile for the most recent run |
| `GET` | `/api/results` | Full ranked candidate list with sub-scores and reasoning |
| `GET` | `/api/download` | Download `submission.csv` |

Interactive API docs are auto-generated at **http://localhost:8000/docs**.

Only one run executes at a time — a second `POST /api/run` while a run is in progress returns `409 Conflict`. This is a deliberate constraint for the single-machine hackathon deployment; see the design note in `run_state.py` for how to extend it to concurrent multi-tenant runs.

---

## Running via CLI only (no API)

The original CLI entry point still works standalone, useful for CI or scripted runs without the dashboard:

```bash
cd backend
python main.py \
  --jd ../data/job_description.docx \
  --candidates ../data/candidates.jsonl \
  --output ../output/submission.csv
```

**Optional CLI flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--faiss-k` | 2000 | Candidate pool size retrieved from FAISS |
| `--top-n` | 100 | Final ranked candidates written to CSV |
| `--force-reembed` | off | Ignore embedding cache and re-encode |
| `--no-lgbm` | off | Skip LightGBM re-ranking |
| `--verbose` | off | Enable DEBUG logging |

On re-runs with unchanged data, embeddings and the FAISS index are loaded from cache — total runtime drops to under 30 seconds. This cache is shared between the CLI and the API server since both read `config.py`'s `OUTPUT_DIR`.

---

## Architecture

**JD Parsing.** `jd_parser.py` reads `job_description.docx` with `python-docx` and uses spaCy (`en_core_web_sm`) plus compiled regex patterns to extract a structured `JDProfile`: role title, seniority band, required years of experience, mandatory skills, preferred skills, and locations. This structured profile drives every downstream scoring function.

**Semantic Retrieval.** `embedder.py` encodes all 100k candidate profiles into L2-normalised vectors using `BAAI/bge-large-en-v1.5`, batched at 256 sequences. Embeddings are cached to `candidate_embeddings.npy` so re-runs skip encoding entirely. The JD is embedded as a single rich string (title + skills + experience) and used to query a `faiss.IndexFlatIP` index — exact inner-product search over unit vectors is equivalent to cosine similarity and runs in under 100ms at this scale. The top-K results form the re-ranking pool.

**Hybrid Scoring.** Each FAISS candidate passes through `scorer.py`, which computes seven sub-scores and combines them as a weighted sum. Experience fit uses a sigmoid curve with three zones (under-, in-, and over-qualified). Career history scores role-tier progression and company quality. A profile quality multiplier dampens candidates with keyword-stuffed or inactive profiles regardless of their semantic score. An optional LightGBM re-ranker in `ranker.py` can apply learned non-linear interactions on top of the hybrid score when training labels are available.

**Output.** `reason_generator.py` produces a recruiter-readable string under 100 words for each top-100 candidate using five composable template clauses (fit label, experience, matched skills, behaviour signals, location). `output.py` writes the final CSV and validates it against the submission schema: exactly 100 rows, sequential ranks 1–100, scores in [0, 1], no nulls, unique candidate IDs.

**Frontend & API.** `app.py` wraps the same pipeline functions used by `main.py` behind a FastAPI service. `POST /api/run` starts the pipeline on a background thread (not `asyncio`, since embedding and FAISS work are CPU-bound and would block the event loop) and returns immediately; `pipeline_executor.py` instruments each of the 7 steps and writes progress into a thread-safe `run_state` singleton. The frontend (`frontend/index.html`) polls `GET /api/status` every 1.5 seconds, renders live per-step progress, then fetches `GET /api/results` once the run completes — no page reload, no hardcoded data.

---

## Design Decisions

**BGE over other embedders.** `bge-large-en-v1.5` consistently tops the MTEB retrieval leaderboard and is designed for asymmetric search (short query → long document), which matches the JD-to-profile retrieval task. It runs on CPU without quantisation at ~500 candidates/second on modest hardware.

**FAISS `IndexFlatIP` over approximate indices.** At 100k vectors × 1024 dimensions, an exact flat index uses ~400MB RAM and searches in under 100ms — no approximation error, no `nprobe` / `ef_search` tuning. Approximate indices (IVF, HNSW) add complexity for negligible gains at this scale.

**Template reasons over LLM.** Calling an LLM for 100 reasoning strings at inference time would add 2–10 minutes and require an API connection, violating both the runtime and offline constraints. Template-based generation is deterministic, consistent in tone, and completes in milliseconds.

**Hybrid score over pure semantic.** Semantic similarity alone ranks candidates who write profiles similar to the JD, not necessarily the most qualified. Explicit skill matching, experience fit, and behaviour signals surface candidates who may have diverse writing styles but strong objective qualifications.

---

## Scoring Weights

| Signal | Weight | Source |
|---|---|---|
| Semantic similarity | **0.30** | BGE cosine vs JD embedding |
| Skill match | **0.20** | Mandatory skill set intersection |
| Experience fit | **0.15** | Sigmoid-normalised YoE vs JD range |
| Career history | **0.10** | Role progression + company tier |
| Behaviour | **0.10** | Response rate, activity, GitHub, offer signals |
| Location | **0.10** | Exact / regional / remote match |
| Education | **0.05** | Degree tier (PhD → BTech → Other) |

> Profile quality (gap penalties, keyword stuffing, inactivity) acts as a multiplicative dampener on the final score rather than an additive term.

---

## Expected Runtime

| Step | Operation | Estimated Time |
|---|---|---|
| 1 | JD parsing (spaCy + regex) | < 5s |
| 2 | Candidate embedding — 100k profiles | 3–4 min (first run) / **< 5s** (cached) |
| 3 | FAISS index build | ~10s (first run) / **< 2s** (cached) |
| 4 | JD embedding + FAISS search (top-2000) | < 5s |
| 5 | Feature extraction for 2000 candidates | ~15s |
| 6 | Scoring + ranking | < 5s |
| 7 | Reason generation + CSV write + validate | < 5s |
| **Total (cold)** | | **~4.5 min** |
| **Total (warm — cache hit)** | | **< 45s** |
