"""Central configuration for RAG-4i-Cloud (Phase 1: local only).

All values are loaded from environment variables with local defaults
matching the original RAG-4i behaviour. A `.env` file in the project
root is supported via the standard library only (no extra dependency).

Required keys (see `.env.example`):
    APP_ENV, VECTOR_STORE, CHROMA_PATH, EMBEDDING_MODEL,
    LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE

PostgreSQL (Phase 2, optional; used only when VECTOR_STORE=postgres):
    DATABASE_URL (takes precedence if set) or
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
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
