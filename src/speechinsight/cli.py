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


@app.command("noise-sweep")
def noise_sweep(
    manifest_path: str = typer.Argument(..., help="JSON manifest of utterances to sweep."),
    model_key: str = typer.Option("asr_base", help="Model key under configs/default.yaml."),
    out_json: str = typer.Option(
        "reports/interim/noise_sweep.json", help="Where to write the sweep points."
    ),
    plot_path: str = typer.Option(
        "reports/interim/noise_sweep.png", help="Where to save the WER-vs-SNR plot."
    ),
) -> None:
    """Run the noise robustness sweep and write both the raw points and the plot."""

    from speechinsight.asr.factory import build_transcriber
    from speechinsight.cache import StageCache
    from speechinsight.data.manifests import read_manifest
    from speechinsight.evaluation.noise_sweep import run_noise_sweep, write_sweep
    from speechinsight.evaluation.plots import plot_noise_sweep

    settings = load_settings()
    corruption = settings.raw.get("corruption", {}).get("noise", {})
    kinds = corruption.get("types", ["white", "pink", "babble"])
    snrs = corruption.get("snr_db", [20, 10, 5, 0])

    manifest = read_manifest(manifest_path)
    transcriber = build_transcriber(settings.raw, model_key)
    cache = StageCache(CONFIG_DIR.parent / settings.paths.cache)

    points = run_noise_sweep(
        transcriber, manifest, model_key, kinds, [float(s) for s in snrs], cache=cache
    )
    write_sweep(points, out_json)
    plot_noise_sweep(points, plot_path)


if __name__ == "__main__":
    app()
