"""Central configuration for RAG-4i-Cloud (Phase 1: local only).

All values are loaded from environment variables with local defaults
matching the original RAG-4i behaviour. A `.env` file in the project
root is supported via the standard library only (no extra dependency).

Required keys (see `.env.example`):
    APP_ENV, VECTOR_STORE, CHROMA_PATH, EMBEDDING_MODEL,
    LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE

Cloud answer LLM (Phase 5A, opt-in; used only when LLM_PROVIDER=bedrock):
    ANSWER_MODEL_ID (Bedrock model or inference-profile ID; required,
    no silent default), BEDROCK_REGION (default us-east-1, in-region)

PostgreSQL (Phase 2, optional; used only when VECTOR_STORE=postgres):
    DATABASE_URL (takes precedence if set) or
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

Document storage (Phase 3, optional; local default, S3 opt-in):
    DOCUMENT_STORAGE, S3_BUCKET, S3_REGION, S3_PREFIX
"""

import os
from dataclasses import dataclass


def _load_dotenv(dotenv_path=".env"):
    """Minimal .env loader using only the stdlib.

    - Ignores blank lines and lines starting with '#'.
    - Supports KEY=VALUE with optional surrounding quotes.
    - Does NOT overwrite variables already present in the environment.
    """
    if not os.path.exists(dotenv_path):
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and (
                    (value[0] == '"' and value[-1] == '"')
                    or (value[0] == "'" and value[-1] == "'")
                ):
                    value = value[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        # A missing/unreadable .env must not break local startup.
        pass


_load_dotenv()


def _get_str(name, default):
    value = os.environ.get(name, default)
    return value if value != "" else default


def _get_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name, default):
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AppConfig:
    app_env: str = "local"
    vector_store: str = "chroma"
    chroma_path: str = "chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_provider: str = "lmstudio"
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "local-model"
    llm_temperature: float = 0.0
    # Cloud answer LLM (Phase 5A). Independent from LLM_MODEL (LM Studio):
    # bedrock uses answer_model_id; credentials always come from the AWS
    # runtime chain, never from config (there is no key setting on purpose).
    answer_model_id: str = ""
    bedrock_region: str = "us-east-1"
    # Query planner / reasoning LLM (Phase 5D). Disabled by default:
    # REASONING_ENABLED=0 reproduces the exact 4.12 behavior. Answer and
    # reasoning models/providers stay independently configurable.
    reasoning_enabled: str = "0"
    reasoning_provider: str = ""
    reasoning_model_id: str = ""
    reasoning_region: str = "us-east-1"
    reasoning_timeout_s: int = 5
    # Answer reviewer / bounded repair (Phase 5E). Disabled by default:
    # REVIEW_ENABLED=0 preserves the pre-5E answer behavior exactly
    # (no reviewer call, no extra initialization). Answer, reasoning,
    # and reviewer providers/models stay independently configurable.
    review_enabled: str = "0"
    review_provider: str = ""
    review_model_id: str = ""
    review_region: str = "us-east-1"
    review_timeout_s: int = 5
    review_always: str = "0"
    # Preserved retrieval/ingestion behaviour (original values).
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    relevance_threshold: float = 0.3
    # PostgreSQL (Phase 2). Defaults point at the SSH tunnel
    # (localhost:15432 -> rag4i-db:5432); RDS itself stays private.
    database_url: str = ""
    db_host: str = "localhost"
    db_port: int = 15432
    db_name: str = "rag4i"
    db_user: str = ""
    db_password: str = ""
    # Document storage (Phase 3). Local default; S3 opt-in via config only.
    document_storage: str = "local"
    s3_bucket: str = ""
    s3_region: str = "ap-south-2"
    s3_prefix: str = "documents/"
    # Session uploads (Phase 5C): local staging root for session files.
    # S3 mode needs no extra path (objects live under sessions/<id>/).
    session_storage_dir: str = "session_uploads"

    def postgres_dsn(self) -> str:
        """Libpq connection string (DATABASE_URL wins if set)."""
        if self.database_url:
            return self.database_url
        from urllib.parse import quote_plus

        user = quote_plus(self.db_user) if self.db_user else ""
        pw = quote_plus(self.db_password) if self.db_password else ""
        auth = ""
        if user and pw:
            auth = f"{user}:{pw}@"
        elif user:
            auth = f"{user}@"
        return f"postgresql://{auth}{self.db_host}:{self.db_port}/{self.db_name}"


_config_cache = None


def load_config() -> AppConfig:
    """Load configuration from environment (with local defaults)."""
    return AppConfig(
        app_env=_get_str("APP_ENV", "local"),
        vector_store=_get_str("VECTOR_STORE", "chroma"),
        chroma_path=_get_str("CHROMA_PATH", "chroma_db"),
        embedding_model=_get_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        llm_provider=_get_str("LLM_PROVIDER", "lmstudio"),
        llm_base_url=_get_str("LLM_BASE_URL", "http://localhost:1234/v1"),
        llm_api_key=_get_str("LLM_API_KEY", "lm-studio"),
        llm_model=_get_str("LLM_MODEL", "local-model"),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.0),
        answer_model_id=_get_str("ANSWER_MODEL_ID", ""),
        bedrock_region=_get_str("BEDROCK_REGION", "us-east-1"),
        reasoning_enabled=_get_str("REASONING_ENABLED", "0"),
        reasoning_provider=_get_str("REASONING_PROVIDER", ""),
        reasoning_model_id=_get_str("REASONING_MODEL_ID", ""),
        reasoning_region=_get_str("REASONING_REGION", "us-east-1"),
        reasoning_timeout_s=_get_int("REASONING_TIMEOUT_S", 5),
        review_enabled=_get_str("REVIEW_ENABLED", "0"),
        review_provider=_get_str("REVIEW_PROVIDER", ""),
        review_model_id=_get_str("REVIEW_MODEL_ID", ""),
        review_region=_get_str("REVIEW_REGION", "us-east-1"),
        review_timeout_s=_get_int("REVIEW_TIMEOUT_S", 5),
        review_always=_get_str("REVIEW_ALWAYS_IF_CONFIGURED", "0"),
        chunk_size=_get_int("CHUNK_SIZE", 1000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 200),
        retrieval_k=_get_int("RETRIEVAL_K", 5),
        relevance_threshold=_get_float("RELEVANCE_THRESHOLD", 0.3),
        database_url=_get_str("DATABASE_URL", ""),
        db_host=_get_str("DB_HOST", "localhost"),
        db_port=_get_int("DB_PORT", 15432),
        db_name=_get_str("DB_NAME", "rag4i"),
        db_user=_get_str("DB_USER", ""),
        db_password=_get_str("DB_PASSWORD", ""),
        document_storage=_get_str("DOCUMENT_STORAGE", "local"),
        s3_bucket=_get_str("S3_BUCKET", ""),
        s3_region=_get_str("S3_REGION", "ap-south-2"),
        s3_prefix=_get_str("S3_PREFIX", "documents/"),
        session_storage_dir=_get_str("SESSION_STORAGE_DIR", "session_uploads"),
    )


def get_config() -> AppConfig:
    """Cached accessor used by the app (re-read via load_config in tests)."""
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def reset_config_cache():
    """Test helper: clear cached config so env changes take effect."""
    global _config_cache
    _config_cache = None
