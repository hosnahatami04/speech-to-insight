from __future__ import annotations

import typer

from speechinsight import __version__
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


if __name__ == "__main__":
    app()
