"""Copy generated PRL figures and data into the manuscript repository.

Resolves ``MANUSCRIPT_DIR`` from ``ldv_analysis.config`` (which reads
``MANUSCRIPT_DIR`` from the repo's ``.env`` file or environment).
Drops ``fig*.{png,pdf,npz}`` and ``fig*.json`` sidecars into
``$MANUSCRIPT_DIR/prl/figures/``.

Plain ``shutil.copy2``, not symlinks: manuscript wants stable bytes
for git history, and Windows symlink semantics across drives are
awkward. Re-runnable on every pipeline change.
"""
from __future__ import annotations

import shutil
import sys

from conventions import MANUSCRIPT_DIR, PIPE_OUT

STEMS = ("fig1", "fig2", "fig2a_5f", "fig3",
         "figS1", "figS1_decoder_check", "figS2")
SUFFIXES = (".png", ".pdf", ".npz", ".json")


def main() -> int:
    if MANUSCRIPT_DIR is None:
        print("MANUSCRIPT_DIR is unset (no .env entry, no env var).",
              file=sys.stderr)
        print("Set MANUSCRIPT_DIR=/path/to/nonlinearphysics-manuscript "
              "in the repo .env file.", file=sys.stderr)
        return 2

    dst_dir = MANUSCRIPT_DIR / "prl" / "figures"
    dst_dir.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_missing = 0
    for stem in STEMS:
        for suffix in SUFFIXES:
            src = PIPE_OUT / f"{stem}{suffix}"
            dst = dst_dir / src.name
            if not src.exists():
                n_missing += 1
                continue
            shutil.copy2(src, dst)
            print(f"  copied {src.name}  ->  {dst}")
            n_copied += 1
    print(f"\n{n_copied} files copied to {dst_dir}, {n_missing} missing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
