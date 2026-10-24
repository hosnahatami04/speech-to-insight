from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class Paths(BaseModel):
    data: Path = Path("data")
    artifacts: Path = Path("artifacts")
    reports: Path = Path("reports")
    cache: Path = Path("artifacts/cache")


class ModelPin(BaseModel):
    """A model identifier pinned to an exact revision or digest so runs are
    reproducible and the resolved weights cannot drift."""

    repo: str
    revision: str


class CorpusPin(BaseModel):
    name: str
    url: str
    license: str
    revision: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPEECHINSIGHT_",
        env_nested_delimiter="__",
    )

    seed: int = 0
    paths: Paths = Field(default_factory=Paths)
    raw: dict[str, Any] = Field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def load_settings(config_dir: Path | None = None) -> Settings:
    """Load ``default.yaml`` and merge any sibling config files, then let
    ``SPEECHINSIGHT_``-prefixed environment variables override top-level keys."""

    base_dir = config_dir or CONFIG_DIR
    merged: dict[str, Any] = {}
    default_path = base_dir / "default.yaml"
    merged.update(_read_yaml(default_path))

    raw: dict[str, Any] = {}
    for yaml_path in sorted(base_dir.glob("*.yaml")):
        if yaml_path.name == "default.yaml":
            continue
        raw[yaml_path.stem] = _read_yaml(yaml_path)

    # YAML supplies defaults, but an environment variable for the same key must
    # win. pydantic-settings ranks init arguments above the env source, so a
    # YAML value is only passed as an init argument when its env var is absent.
    init: dict[str, Any] = {"raw": raw}
    if "seed" in merged and "SPEECHINSIGHT_SEED" not in os.environ:
        init["seed"] = merged["seed"]
    if "paths" in merged and "SPEECHINSIGHT_PATHS" not in os.environ:
        init["paths"] = merged["paths"]
    return Settings(**init)
