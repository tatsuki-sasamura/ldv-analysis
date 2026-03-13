"""Plotting utilities for LDV analysis figures.

Provides shared figure-saving logic and publication style presets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_fig(fig: Figure, name: str, output_dir: str | Path, *,
             dpi: int | None = None, close: bool = True) -> Path:
    """Save a figure to *output_dir* as PNG with consistent settings.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    name : str
        Base filename (without extension).
    output_dir : str or Path
        Directory to write into (created if needed).
    dpi : int or None
        Override DPI; defaults to ``figure.dpi`` rcParam.
    close : bool
        Close the figure after saving (default True).

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi or plt.rcParams["figure.dpi"])
    print(f"  Saved: {path}")
    if close:
        plt.close(fig)
    return path


# Publication-quality rcParams (single-column APS/IEEE style)
_MANUSCRIPT_PARAMS = {
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 0.75,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
}


def apply_manuscript_style() -> None:
    """Apply publication-quality matplotlib rcParams on top of science+ieee."""
    plt.rcParams.update(_MANUSCRIPT_PARAMS)
