#!/bin/bash
# Build the full NEATM lookup tables for the mm/THz range: 50-1500 GHz every 25 GHz plus the ACT band centres
# (90 and 220 GHz), T_ss = 100-1200 K every 10 K, phase angle 0-180 deg every 5 deg.
# Restartable: tables that already exist are skipped. Usage: scripts/build_full_mm_tables.sh [outdir]
# Progress: grep -c "^Done" <outdir>/build.log (the build is complete when this reaches the number of tables, 61).
# <outdir>/DONE is written by this wrapper when the build ends, but in one run it was missing although the
# build had finished (the wrapper was apparently killed at logout), so check the log rather than relying on it.
cd "$(dirname "$0")/.." || exit 1
OUT=${1:-data/mm_tables}
PY=${PYTHON:-/usr/bin/python3.12}
mkdir -p "$OUT"
rm -f "$OUT/DONE"
PYTHONPATH=. nice -n 5 "$PY" scripts/build_mm_tables.py \
    --fmin 50 --fmax 1500 --df 25 --freqs 90 220 \
    --tmin 100 --tmax 1200 --tstep 10 \
    --amin 0 --amax 3.141592653589793 --alpha-step 0.08726646259971647 \
    --threads "${THREADS:-8}" --out "$OUT/" >> "$OUT/build.log" 2>&1
echo "exit status $? at $(date -Is)" > "$OUT/DONE"
