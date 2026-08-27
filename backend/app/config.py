from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_backend: str
    database_url: str | None
    olist_dataset_dir: str | None
    voyage_api_key: str | None
    anthropic_api_key: str | None
    enable_reranking: bool
    retrieval_candidate_depth: int
    retrieval_final_depth: int
    retrieval_mode: str


def _read_env_file() -> dict[str, str]:
    env_path = Path(".env")
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def get_settings() -> Settings:
    env_values = _read_env_file()

    def env_value(name: str, default: str) -> str:
        return os.getenv(name, env_values.get(name, default))

    def env_bool(name: str, default: bool) -> bool:
        return env_value(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

    def env_int(name: str, default: int) -> int:
        try:
            return max(1, int(env_value(name, str(default))))
        except ValueError:
            return default

    return Settings(
        data_backend=os.getenv(
            "SUPPORTBOT_DATA_BACKEND",
            env_values.get("SUPPORTBOT_DATA_BACKEND", "memory"),
        ),
        database_url=os.getenv("DATABASE_URL", env_values.get("DATABASE_URL")),
        olist_dataset_dir=os.getenv(
            "OLIST_DATASET_DIR",
            env_values.get("OLIST_DATASET_DIR"),
        ),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", env_values.get("VOYAGE_API_KEY")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", env_values.get("ANTHROPIC_API_KEY")),
        enable_reranking=env_bool("SUPPORTBOT_ENABLE_RERANKING", False),
        retrieval_candidate_depth=env_int("SUPPORTBOT_RETRIEVAL_CANDIDATE_DEPTH", 20),
        retrieval_final_depth=env_int("SUPPORTBOT_RETRIEVAL_FINAL_DEPTH", 3),
        retrieval_mode=env_value("SUPPORTBOT_RETRIEVAL_MODE", "weighted"),
    )
