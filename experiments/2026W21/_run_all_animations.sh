#!/bin/bash
# Loop pressure_map_2d.py --animate composed over every .h5 in the two
# waveguide-test datasets.  Skips files whose anim MP4 already exists so
# the script is resumable.

set -u
DATA="C:/Users/Tatsuki Sasamura/OneDrive - Lund University/Data/output/W21"
OUT="experiments/2026W21/output"
PY=".venv/Scripts/python"

run_dir () {
    local sd="$1"
    local outdir="$OUT/$(basename "$sd")/pressure_map_2d"
    for h5 in "$sd"/f*.h5; do
        local stem
        stem=$(basename "$h5" .h5)
        for mode in composed 1f 2f; do
            local mp4="$outdir/anim_${mode}_${stem}.mp4"
            if [ -f "$mp4" ]; then
                echo "skip  $(basename "$sd")/$stem  [$mode] (mp4 exists)"
                continue
            fi
            echo "run   $(basename "$sd")/$stem  [$mode]"
            "$PY" experiments/2026W21/pressure_map_2d.py "$h5" --animate "$mode" \
                > /tmp/_anim.log 2>&1 || {
                echo "FAIL  $stem [$mode] — log:"
                tail -20 /tmp/_anim.log
            }
        done
    done
}

run_dir "$DATA/sample_101x77_fsweep_1p89to1p92_1kHz_60Vpp_20260530_031237"
run_dir "$DATA/sample_101x77_fsweep_3p7to3p9_60Vpp_20260530_013206"

echo "=== ALL DONE ==="
