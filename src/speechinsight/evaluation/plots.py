from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from speechinsight.evaluation.accents import AccentWer
from speechinsight.evaluation.noise_sweep import SweepPoint


def _use_agg_backend() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)


def plot_noise_sweep(points: list[SweepPoint], out_path: str | Path) -> Path:
    """Draw one WER-vs-SNR curve per (model, noise_kind) pair. Error bars show
    the bootstrap CI. Matplotlib is imported lazily so the module stays usable
    on a tester machine without the plotting group installed."""

    _use_agg_backend()
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    grouped: dict[tuple[str, str], list[SweepPoint]] = defaultdict(list)
    for p in points:
        grouped[(p.model, p.noise_kind)].append(p)

    for (model, kind), pts in sorted(grouped.items()):
        pts = sorted(pts, key=lambda x: x.snr_db)
        snrs = [p.snr_db for p in pts]
        wers = [p.wer for p in pts]
        errs = [
            [p.wer - p.ci_low for p in pts],
            [p.ci_high - p.wer for p in pts],
        ]
        ax.errorbar(snrs, wers, yerr=errs, marker="o", label=f"{model} · {kind}")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("WER")
    ax.set_title("Speech recognition under added noise")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # lower SNR (harder) on the right
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_per_accent_wer(results: list[AccentWer], out_path: str | Path) -> Path:
    """Horizontal bar chart of WER per accent with CI error bars. Accents with
    fewer than the conclusion threshold are drawn in a lighter color so the
    reader does not weight them equally."""

    _use_agg_backend()
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = sorted(results, key=lambda r: r.wer)
    labels = [f"{r.accent} (n={r.utterances})" for r in results]
    wers = [r.wer for r in results]
    err_low = [r.wer - r.ci_low for r in results]
    err_high = [r.ci_high - r.wer for r in results]
    colors = ["#888" if r.note else "#2a6" for r in results]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(results) + 1)))
    ax.barh(labels, wers, xerr=[err_low, err_high], color=colors)
    ax.set_xlabel("WER")
    ax.set_title("Per-accent WER (light bars are report-only, under threshold)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
