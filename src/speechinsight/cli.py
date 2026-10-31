from __future__ import annotations

import typer

from speechinsight import __version__
from speechinsight.config import CONFIG_DIR, load_settings
from speechinsight.logging import setup_logging

app = typer.Typer(
    name="speechinsight",
    add_completion=False,
    help="From meeting audio to grounded decisions and action items.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed version and exit.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
) -> None:
    setup_logging(log_level)


@app.command()
def data() -> None:
    """Build corpus manifests and the data card from locally available files."""

    from speechinsight.data.build import build_all

    settings = load_settings()
    data_cfg = settings.raw.get("data", {})
    root = CONFIG_DIR.parent
    reports_dir = root / settings.paths.reports
    build_all(root, data_cfg, reports_dir)


if __name__ == "__main__":
    app()
