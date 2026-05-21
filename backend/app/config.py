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
    )
