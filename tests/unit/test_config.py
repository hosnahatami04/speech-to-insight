from __future__ import annotations

from pathlib import Path

from speechinsight.config import load_settings


def _write_config(config_dir: Path) -> None:
    (config_dir / "default.yaml").write_text("seed: 7\npaths:\n  data: data\n", encoding="utf-8")
    (config_dir / "asr.yaml").write_text("beam_size: 5\n", encoding="utf-8")


def test_loads_default_and_sibling_configs(tmp_path: Path) -> None:
    _write_config(tmp_path)
    settings = load_settings(tmp_path)
    assert settings.seed == 7
    assert settings.raw["asr"]["beam_size"] == 5


def test_env_overrides_top_level_key(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("SPEECHINSIGHT_SEED", "42")
    settings = load_settings(tmp_path)
    assert settings.seed == 42


def test_missing_config_dir_yields_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "absent")
    assert settings.seed == 0
    assert settings.raw == {}
