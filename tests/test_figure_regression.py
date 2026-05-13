"""Regression tests for manuscript and AF2026 figure NPZ caches.

Compares the latest figure NPZ produced by ``manuscript_figures.py`` and
``af2026_figures.py`` (in ``$MANUSCRIPT_DIR``) against a frozen baseline
committed inside the repo at ``tests/fixtures/figures/``.

These tests are marked ``slow`` and **deselected by default** in
``pyproject.toml`` so CI stays fast. Run them explicitly:

    pytest -m slow tests/test_figure_regression.py

The tests skip cleanly when ``MANUSCRIPT_DIR`` is unset or the live
output is absent — no failure on a fresh clone.

Workflow when you intentionally change a figure
-----------------------------------------------
1. Regenerate locally:
       python experiments/2026W10_stepA/manuscript_figures.py --fresh
       python experiments/2026W10_stepA/af2026_figures.py
2. Confirm the new output is correct by eye.
3. Update the in-repo baseline:
       cp $MANUSCRIPT_DIR/pra/figures/FigN.npz tests/fixtures/figures/FigN.npz
       (etc.)
4. Commit the baseline alongside the change.
"""

import os
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "figures"


def _manuscript_dir() -> Path | None:
    """Resolve ``MANUSCRIPT_DIR`` from env or the repo's ``.env`` file."""
    val = os.environ.get("MANUSCRIPT_DIR")
    if not val:
        env_file = Path(__file__).resolve().parents[1] / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                if line.startswith("MANUSCRIPT_DIR="):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    break
    return Path(val) if val else None


# (baseline-fixture-name, live-output-path-template) pairs.
# Path template is relative to MANUSCRIPT_DIR; {key} is the file's stem.
_FIGURES = [
    ("Fig5",  "pra/figures/Fig5.npz"),
    ("Fig6",  "pra/figures/Fig6.npz"),
    ("Fig7",  "pra/figures/Fig7.npz"),
    ("Fig8",  "pra/figures/Fig8.npz"),
    ("FigA1", "pra/figures/FigA1.npz"),
    ("AF_Fig1", "acoustofluidics/figures/Fig1.npz"),
    ("AF_Fig2", "acoustofluidics/figures/Fig2.npz"),
]


def _live_path(rel: str) -> Path | None:
    base = _manuscript_dir()
    if base is None:
        return None
    p = base / rel
    return p if p.exists() else None


def _compare_npz(live: Path, baseline: Path) -> list[str]:
    """Return a list of human-readable problem descriptions; empty = OK.

    Compares every numeric array in the two .npz files. Tolerance is
    tight (rtol=1e-9, atol=1e-12) because both should come from
    deterministic computations on the same TDMS data.
    """
    problems: list[str] = []
    a_npz = np.load(live, allow_pickle=False)
    b_npz = np.load(baseline, allow_pickle=False)
    a_keys, b_keys = set(a_npz.files), set(b_npz.files)
    if a_keys != b_keys:
        only_live = sorted(a_keys - b_keys)
        only_base = sorted(b_keys - a_keys)
        if only_live:
            problems.append(f"keys only in live: {only_live}")
        if only_base:
            problems.append(f"keys only in baseline: {only_base}")
    for k in sorted(a_keys & b_keys):
        a = a_npz[k]
        b = b_npz[k]
        if a.dtype.kind in {"U", "S", "O"}:
            # String / object — compare equality
            if not np.array_equal(a, b):
                problems.append(f"{k!r}: string/object value differs")
            continue
        if a.shape != b.shape:
            problems.append(f"{k!r}: shape {a.shape} vs {b.shape}")
            continue
        if not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            max_abs = float(np.nanmax(np.abs(a.astype(float)
                                              - b.astype(float))))
            problems.append(f"{k!r}: max |Δ|={max_abs:.3e}")
    return problems


@pytest.mark.slow
@pytest.mark.parametrize("baseline_name,rel_path", _FIGURES)
def test_figure_npz_matches_baseline(baseline_name, rel_path):
    """Each manuscript figure's NPZ matches the committed baseline."""
    baseline = FIXTURES / f"{baseline_name}.npz"
    if not baseline.exists():
        pytest.skip(f"baseline missing: {baseline}")
    live = _live_path(rel_path)
    if live is None:
        pytest.skip(f"live output not found: {rel_path} "
                    f"(set MANUSCRIPT_DIR and regenerate figures)")
    problems = _compare_npz(live, baseline)
    assert not problems, (
        f"{baseline_name} differs from baseline:\n  "
        + "\n  ".join(problems)
    )
