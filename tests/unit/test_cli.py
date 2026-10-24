from __future__ import annotations

from typer.testing import CliRunner

from speechinsight import __version__
from speechinsight.cli import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__
