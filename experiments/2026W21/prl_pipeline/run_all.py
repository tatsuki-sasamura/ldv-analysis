"""Regenerate every PRL figure (Fig 1, 2, 3, S1, S2) in sequence.

Usage::

    python experiments/2026W21/prl_pipeline/run_all.py [--skip figS2]

The order matters slightly: figS1 cross-reads ``fig2.npz`` for the
spectral-vs-time P_in scatter, so fig2 must run before figS1.
figS2 is independent.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Each entry: (name, importable module, callable taking no args)
STEPS = [
    ("fig1", "fig1", "main"),
    ("fig2", "fig2", "main"),
    ("fig3", "fig3", "main"),
    ("figS1", "figS1", "main"),
    ("figS2", "figS2", "main"),
]


def _run_step(_name: str, module_name: str, fn_name: str) -> tuple[bool, float]:
    mod = __import__(module_name)
    fn = getattr(mod, fn_name)
    t0 = time.perf_counter()
    try:
        fn()
        return True, time.perf_counter() - t0
    except Exception:
        traceback.print_exc()
        return False, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip", action="append", default=[],
                    help="Skip a step by name (repeatable). E.g. --skip figS2.")
    ap.add_argument("--only", action="append", default=[],
                    help="Run only these steps by name (repeatable).")
    args = ap.parse_args()

    only = set(args.only)
    skip = set(args.skip)

    print(f"PRL pipeline: {len(STEPS)} steps")
    results = []
    for name, module_name, fn_name in STEPS:
        if only and name not in only:
            print(f"\n--- SKIP {name} (not in --only) ---")
            continue
        if name in skip:
            print(f"\n--- SKIP {name} (in --skip) ---")
            continue
        print(f"\n=== {name} ===")
        ok, dt = _run_step(name, module_name, fn_name)
        results.append((name, ok, dt))
        print(f"  {name}: {'OK' if ok else 'FAIL'} in {dt:.1f} s")

    print("\n=== summary ===")
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, dt in results:
        print(f"  {name:>8}: {'OK' if ok else 'FAIL':>4}  {dt:6.1f} s")
    print(f"  {n_ok}/{len(results)} steps succeeded")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
