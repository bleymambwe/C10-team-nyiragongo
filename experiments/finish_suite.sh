#!/bin/bash
# Complete the outstanding experiments, in order of scientific value.
# Each writes incremental checkpoints, so stopping loses at most one unit.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=3

echo "=== E4: active design (closes prediction P5) ==="
python -u experiments/e4_active_design.py 2>&1 | grep -v "it/s" | tail -30
echo "E4 exit=$?"

echo "=== E5: complete the third model (resume) ==="
python -u experiments/e5_real_data_multimodel.py 2>&1 | grep -v "it/s" | tail -25
echo "E5 exit=$?"

echo "=== E2: trained toy transformers + confound dial ==="
python -u experiments/e2_trained_toy.py 2>&1 | grep -v "it/s" | tail -40
echo "E2 exit=$?"

echo "ALL_REMAINING_DONE"
ls -la results/data/
