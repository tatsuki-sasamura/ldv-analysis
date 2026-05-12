# %%
"""Convert a v1 TDMS scan to the v2 HDF5 schema.

Bridges the broken-setup acquisitions to the format the rebuilt DAQ
will produce. Reads a TDMS via ``load_scan_tdms``, augments the
metadata with the v2-required fields not present in the original file
(chip_id, session_id, burst timing — taken from the FFT cache or CLI
overrides), and writes via ``write_scan_hdf5``.

Usage:
    python experiments/2026W17_newsetup/tdms_to_hdf5.py \\
        "$LDV_DATA_ROOT/20260307experimentB/test10_1907_5Vpp_1m_s_max.tdms" \\
        out.h5 --chip-id ldv_chip_2026_W10 --session-id 20260307_test10

    # smaller test fixture: keep only the first 100 scan points
    python ... --max-points 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from ldv_analysis.io_utils import (
    ROLE_DRIVE_VOLTAGE, ROLE_LDV_OUTPUT, ROLE_CURRENT,
    ScanData, load_scan_tdms, write_scan_hdf5,
)


def _take_first_n_points(scan: ScanData, n: int) -> ScanData:
    """Return a ScanData view limited to the first ``n`` points.

    Used for ``--max-points`` to produce smaller fixtures. The lazy
    loader is wrapped so any caller still gets ``(k, n_samples)`` arrays.
    """
    inner = scan._loader
    new_pos_x = scan.pos_x[:n]
    new_pos_y = scan.pos_y[:n]
    new_rssi = scan.rssi[:n] if scan.rssi is not None else None

    def loader(role, points):
        if isinstance(points, slice):
            idx = np.arange(*points.indices(n))
        else:
            idx = np.asarray(points, dtype=int)
            if (idx < 0).any() or (idx >= n).any():
                raise IndexError("point index out of range")
        return inner(role, idx)

    return ScanData(
        pos_x=new_pos_x, pos_y=new_pos_y, rssi=new_rssi,
        dt=scan.dt, n_points=n, n_samples=scan.n_samples,
        metadata=dict(scan.metadata),
        _loader=loader,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tdms_path", type=Path)
    p.add_argument("output", type=Path,
                   help="HDF5 output path (.h5)")
    p.add_argument("--chip-id", required=True,
                   help="Chip identifier (links to chip-cal sidecar JSON)")
    p.add_argument("--session-id", required=True,
                   help="Session identifier (groups runs under one calibration)")
    p.add_argument("--burst-on-us", type=float, default=5.0,
                   help="Nominal burst ON time in microseconds (default: 5)")
    p.add_argument("--burst-off-us", type=float, default=525.0,
                   help="Nominal burst OFF time in microseconds (default: 525)")
    p.add_argument("--operator", default="",
                   help="Operator name for provenance")
    p.add_argument("--notes", default="",
                   help="Free-text provenance notes")
    p.add_argument("--max-points", type=int, default=None,
                   help="If set, write only the first N scan points "
                        "(useful for small test fixtures)")
    p.add_argument("--chunk-points", type=int, default=100,
                   help="Streaming chunk size for waveform writes (default: 100)")
    args = p.parse_args()

    if not args.tdms_path.exists():
        sys.exit(f"TDMS not found: {args.tdms_path}")
    if args.output.exists():
        sys.exit(f"Refusing to overwrite existing {args.output}; "
                 f"delete it first.")

    print(f"Reading: {args.tdms_path.name}")
    scan = load_scan_tdms(args.tdms_path)

    print(f"  {scan.n_points} points x {scan.n_samples} samples, "
          f"dt = {scan.dt*1e9:.1f} ns")
    print(f"  roles available: {scan.metadata['_available_roles']}")

    if args.max_points is not None and args.max_points < scan.n_points:
        print(f"  trimming to first {args.max_points} points "
              f"(was {scan.n_points})")
        scan = _take_first_n_points(scan, args.max_points)

    # Augment metadata with v2-required fields not in v1 TDMS
    import datetime
    scan.metadata.update({
        "chip_id": args.chip_id,
        "session_id": args.session_id,
        "burst_on_us_nominal": args.burst_on_us,
        "burst_off_us_nominal": args.burst_off_us,
        "operator": args.operator,
        "notes": args.notes,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "daq_software_version": "tdms_to_hdf5_converter",
        "source_v1_tdms": args.tdms_path.name,
    })

    # Channels that aren't drive/ldv/current we drop on this path
    available_roles = scan.metadata["_available_roles"]
    canonical = {ROLE_DRIVE_VOLTAGE, ROLE_LDV_OUTPUT, ROLE_CURRENT}
    scan.metadata["_available_roles"] = sorted(
        r for r in available_roles if r in canonical
    )

    print(f"Writing: {args.output}")
    write_scan_hdf5(scan, args.output, chunk_points=args.chunk_points)

    size_mb = args.output.stat().st_size / 1e6
    print(f"  Done: {args.output.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
