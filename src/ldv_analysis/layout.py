"""Layout-aware figure sizing for multi-panel matplotlib figures.

Measures tight_layout overhead (margins, gaps) from reference figures so
that ``figsize_for_layout(n_rows, n_cols)`` gives each subplot the same
axes area as a single-panel figure at the style's default figsize.

Model::

    fig_h = margin_v + n_rows * ax_h + (n_rows - 1) * gap_v
    fig_w = margin_h + n_cols * ax_w + (n_cols - 1) * gap_h

Gaps differ between shared and non-shared axes because ``sharex`` removes
redundant x-labels (reducing vertical gap) and ``sharey`` removes
redundant y-labels (reducing horizontal gap).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Representative labels matching the analysis scripts
_XLABEL = "X position (mm)"
_YLABEL = "Y value (unit)"
_TITLE = "Title text"


def _measure_layout_overhead() -> tuple[
    float, float, float, float, float, float, float, float
]:
    """Measure tight_layout margins and gaps from reference figures.

    Returns
    -------
    ref_ax_w, ref_ax_h : float
        Reference axes size (inches) for a 1x1 figure.
    margin_h, margin_v : float
        Total horizontal/vertical margin (inches).
    gap_h_shared, gap_h_indep : float
        Horizontal gap between columns with/without shared y-axes.
    gap_v_shared, gap_v_indep : float
        Vertical gap between rows with/without shared x-axes.
    """
    _w, _h = plt.rcParams["figure.figsize"]
    _x = np.linspace(0, 1, 10)
    _y = np.zeros(10)

    def _axes_size(fig, ax):
        bbox = ax.get_position()
        return bbox.width * fig.get_figwidth(), bbox.height * fig.get_figheight()

    # 1x1 reference: total margin
    fig, ax = plt.subplots(1, 1, figsize=(_w, _h))
    ax.plot(_x, _y)
    ax.set_xlabel(_XLABEL); ax.set_ylabel(_YLABEL); ax.set_title(_TITLE)
    fig.tight_layout()
    ref_w, ref_h = _axes_size(fig, ax)
    plt.close(fig)

    margin_h = _w - ref_w
    margin_v = _h - ref_h

    # 2x1 sharex=True: vertical gap (shared)
    fig, axes = plt.subplots(2, 1, figsize=(_w, 2 * _h), sharex=True)
    for a in axes:
        a.plot(_x, _y); a.set_ylabel(_YLABEL)
    axes[-1].set_xlabel(_XLABEL); axes[0].set_title(_TITLE)
    fig.tight_layout()
    _, h2_shared = _axes_size(fig, axes[0])
    plt.close(fig)
    gap_v_shared = 2 * _h - margin_v - 2 * h2_shared

    # 2x1 sharex=False: vertical gap (independent)
    fig, axes = plt.subplots(2, 1, figsize=(_w, 2 * _h), sharex=False)
    for a in axes:
        a.plot(_x, _y); a.set_ylabel(_YLABEL); a.set_xlabel(_XLABEL)
    axes[0].set_title(_TITLE)
    fig.tight_layout()
    _, h2_indep = _axes_size(fig, axes[0])
    plt.close(fig)
    gap_v_indep = 2 * _h - margin_v - 2 * h2_indep

    # 1x2 sharey=True: horizontal gap (shared)
    fig, axes = plt.subplots(1, 2, figsize=(2 * _w, _h), sharey=True)
    for a in axes:
        a.plot(_x, _y); a.set_xlabel(_XLABEL)
    axes[0].set_ylabel(_YLABEL); axes[0].set_title(_TITLE)
    fig.tight_layout()
    w2_shared, _ = _axes_size(fig, axes[0])
    plt.close(fig)
    gap_h_shared = max(2 * _w - margin_h - 2 * w2_shared, 0)

    # 1x2 sharey=False: horizontal gap (independent)
    fig, axes = plt.subplots(1, 2, figsize=(2 * _w, _h), sharey=False)
    for a in axes:
        a.plot(_x, _y); a.set_xlabel(_XLABEL); a.set_ylabel(_YLABEL)
    axes[0].set_title(_TITLE)
    fig.tight_layout()
    w2_indep, _ = _axes_size(fig, axes[0])
    plt.close(fig)
    gap_h_indep = max(2 * _w - margin_h - 2 * w2_indep, 0)

    return (ref_w, ref_h, margin_h, margin_v,
            gap_h_shared, gap_h_indep, gap_v_shared, gap_v_indep)


(_REF_AX_W, _REF_AX_H, _MARGIN_H, _MARGIN_V,
 _GAP_H_SHARED, _GAP_H_INDEP, _GAP_V_SHARED, _GAP_V_INDEP,
 ) = _measure_layout_overhead()


def figsize_for_layout(
    n_rows: int = 1,
    n_cols: int = 1,
    ax_w_scale: float = 1.0,
    ax_h_scale: float = 1.0,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[float, float]:
    """Compute figsize so each subplot's axes area matches the 1x1 reference.

    Parameters
    ----------
    n_rows, n_cols : int
        Subplot grid dimensions.
    ax_w_scale, ax_h_scale : float
        Multiplier on the reference axes width/height (e.g. 2.0 for a
        double-wide single plot).
    sharex, sharey : bool
        Whether axes sharing is used (affects inter-subplot gaps).
    """
    target_w = ax_w_scale * _REF_AX_W
    target_h = ax_h_scale * _REF_AX_H
    gap_h = _GAP_H_SHARED if sharey else _GAP_H_INDEP
    gap_v = _GAP_V_SHARED if sharex else _GAP_V_INDEP
    fig_w = _MARGIN_H + n_cols * target_w + max(n_cols - 1, 0) * gap_h
    fig_h = _MARGIN_V + n_rows * target_h + max(n_rows - 1, 0) * gap_v
    return (fig_w, fig_h)
