# config.py — Central configuration for the AI Candidate Ranking System

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
# backend/config.py lives inside <project_root>/backend/, so PROJECT_ROOT is one
# level up. data/, output/, and models/ are siblings of backend/, not nested
# inside it — this lets the FastAPI app, the CLI (main.py), and any notebook
# all resolve the same absolute paths regardless of current working directory.
BASE_DIR     = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "output"

# ── Input files ───────────────────────────────────────────────────────────────
JD_PATH             = DATA_DIR / "job_description.docx"
CANDIDATES_PATH     = DATA_DIR / "candidates.jsonl"
SCHEMA_PATH         = DATA_DIR / "candidate_schema.json"
SIGNALS_DOC_PATH    = DATA_DIR / "redrob_signals_doc.docx"

# ── Output files ──────────────────────────────────────────────────────────────
SUBMISSION_PATH     = OUTPUT_DIR / "submission.csv"
FAISS_INDEX_PATH    = OUTPUT_DIR / "candidate_index.faiss"
EMBEDDINGS_PATH     = OUTPUT_DIR / "candidate_embeddings.npy"

# ── Models (all offline / local) ──────────────────────────────────────────────
EMBEDDING_MODEL     = "BAAI/bge-small-en-v1.5"   # fast CPU-friendly BGE variant
SPACY_MODEL         = "en_core_web_sm"
LIGHTGBM_MODEL_PATH = PROJECT_ROOT / "models" / "ranker.lgb"  # trained artifact

# ── Retrieval settings ────────────────────────────────────────────────────────
FAISS_TOP_K         = 500    # coarse FAISS recall before re-ranking
FINAL_TOP_N         = 100    # candidates written to submission.csv

# ── Runtime constraints ───────────────────────────────────────────────────────
MAX_RUNTIME_SECONDS = 270    # stay well under the 5-min hard cap
BATCH_SIZE          = 512   # embedding batch size; bge-small is light enough on CPU
                              # that 512 cuts wall-clock encode time vs. 256 with no
                              # accuracy change — lower this back down if you hit RAM limits
RANDOM_SEED         = 42

# ── Hybrid scoring weights ────────────────────────────────────────────────────
# Must sum to 1.0
WEIGHTS: dict[str, float] = {
    "semantic"       : 0.30,   # BGE cosine similarity vs JD embedding
    "skill_match"    : 0.20,   # mandatory + preferred skill overlap
    "experience"     : 0.15,   # years-of-experience fit
    "career_history" : 0.10,   # progression signals (title trajectory, gaps)
    "behaviour"      : 0.10,   # redrob behavioural signals
    "education"      : 0.05,   # degree / institution tier
    "location"       : 0.10,   # geo match / relocation flag
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"

# ── Seniority band → expected YoE mapping ────────────────────────────────────
SENIORITY_YOE: dict[str, tuple[int, int]] = {
    "junior" : (0,  3),
    "mid"    : (3,  6),
    "senior" : (6, 12),
    "lead"   : (10, 99),
}

# ── Tech-keyword taxonomy (used by JD parser & skill scorer) ─────────────────
TECH_TAXONOMY: list[str] = [
    # Languages
    "python", "java", "scala", "kotlin", "go", "golang", "rust", "c++", "c#",
    "javascript", "typescript", "r", "sql", "bash", "shell",
    # ML / Data
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "lightgbm",
    "huggingface", "transformers", "spacy", "nltk", "opencv",
    "pandas", "numpy", "scipy", "polars",
    # Infra / MLOps
    "docker", "kubernetes", "airflow", "mlflow", "kubeflow", "ray",
    "spark", "kafka", "flink", "dbt",
    "aws", "gcp", "azure", "s3", "gcs",
    "faiss", "elasticsearch", "pinecone", "weaviate", "chroma",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "cassandra", "bigquery",
    "snowflake", "databricks",
    # Web / API
    "fastapi", "flask", "django", "graphql", "rest", "grpc",
    # Practices
    "git", "ci/cd", "github actions", "terraform", "ansible",
    "agile", "scrum", "tdd",
]
