#!/usr/bin/env python3
"""
Scatter plot: mean KL divergence vs. on-disk model size.

Usage
-----
From the PlottingScripts directory (or pass full paths):

    python plot_kld_vs_size.py
    python plot_kld_vs_size.py --show
    python plot_kld_vs_size.py -o kld_vs_size.png
    python plot_kld_vs_size.py --data quants.json -o kld_vs_size.png --show

Arguments
---------
--data PATH
    JSON file with quant definitions (default: quants.json next to this script).
    Expected top-level keys: ``title``, ``subtitle``, and ``quants`` (a list of
    objects with ``id``, ``creator``, ``type``, ``disk_size_gib``, ``mean_kld``).

-o, --output PATH
    Save the plot as a PNG at PATH. If omitted, no file is written unless you
    use ``--show``.

--show
    Open an interactive matplotlib window. Without ``--output`` or ``--show``,
    the script loads data and exits without displaying or saving a plot.

Plot behavior
-------------
Marker shape is determined by quant *type*; color by *creator*. The legend
(upper right) lists each quant with its mean KLD, sorted worst-to-best (highest
KLD at top, lowest at bottom); the plot shows only markers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Cream-style background similar to reference charts
FIG_FACE = "#F5F5E1"
AX_FACE = "#F5F5E1"
GRID_COLOR = "#c8c8b8"

# Prefer explicit markers for known types so NVFP4 is always the same shape.
TYPE_MARKERS: dict[str, str] = {
    "INT8": "o",
    "NVFP4": "^",
    "AWQ-INT4": "s",
    "AWQ-BF16-INT4": "D",
}
_FALLBACK_MARKERS = ("v", "P", "*", "X", "h", "8", "d", "p")


def _load_quants(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _creator_colors(creators: list[str]) -> dict[str, str]:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    unique = sorted(set(creators))
    return {c: palette[i % len(palette)] for i, c in enumerate(unique)}


def _type_markers(types: list[str]) -> dict[str, str]:
    unique = sorted(set(types))
    out: dict[str, str] = {}
    fallback_i = 0
    for t in unique:
        if t in TYPE_MARKERS:
            out[t] = TYPE_MARKERS[t]
        else:
            out[t] = _FALLBACK_MARKERS[fallback_i % len(_FALLBACK_MARKERS)]
            fallback_i += 1
    return out


def plot_from_payload(
    payload: dict[str, Any],
    *,
    outfile: Path | None = None,
    show: bool = False,
) -> None:
    quants: list[dict[str, Any]] = payload["quants"]
    title = payload.get("title", "")
    subtitle = payload.get("subtitle", "")

    creators = [str(q["creator"]) for q in quants]
    types = [str(q["type"]) for q in quants]
    c_map = _creator_colors(creators)
    m_map = _type_markers(types)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=FIG_FACE)
    ax.set_facecolor(AX_FACE)

    xs = [float(q["disk_size_gib"]) for q in quants]
    ys = [float(q["mean_kld"]) for q in quants]

    for q, x, y in zip(quants, xs, ys, strict=True):
        creator = str(q["creator"])
        qtype = str(q["type"])
        ax.scatter(
            [x],
            [y],
            s=120,
            c=c_map[creator],
            marker=m_map[qtype],
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )

    ax.set_xlabel("File Size (GiB)", fontweight="bold")
    ax.set_ylabel("Mean KL Divergence (lower is better)", fontweight="bold")
    title_parts = [t for t in (title, subtitle) if t]
    if title_parts:
        ax.set_title(
            "\n".join(title_parts),
            fontweight="bold",
            fontsize=11,
            pad=16,
        )

    ax.grid(True, linestyle=":", linewidth=0.8, color=GRID_COLOR, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")

    # Padding so points and legend are not clipped
    x_pad = max(1.0, (max(xs) - min(xs)) * 0.12 + 0.5)
    y_pad = max(0.02, (max(ys) - min(ys)) * 0.15 + 0.01)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(max(0.0, min(ys) - y_pad), max(ys) + y_pad)

    legend_handles: list[Line2D] = []
    for q in sorted(quants, key=lambda q: float(q["mean_kld"]), reverse=True):
        creator = str(q["creator"])
        qtype = str(q["type"])
        qid = str(q.get("id", f'{creator}/{q.get("model", "")}'))
        kld = float(q["mean_kld"])
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=c_map[creator],
                marker=m_map[qtype],
                linestyle="None",
                markersize=9,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=f"{qid} — KLD {kld:.6f}",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=8,
    )

    fig.tight_layout()
    if outfile:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, facecolor=FIG_FACE, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=here / "quants.json",
        help="Path to JSON quant definitions",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write PNG to this path (default: no file)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window",
    )
    args = parser.parse_args()

    payload = _load_quants(args.data)
    plot_from_payload(payload, outfile=args.output, show=args.show)


if __name__ == "__main__":
    main()
