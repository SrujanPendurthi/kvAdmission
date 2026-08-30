#!/usr/bin/env bash
# Run every pending experiment, pull-drain, then power off the box.
#
# Sequenced cheapest-first so partial results are still useful if anything
# dies partway. Each writes its own results file and is independently
# restartable.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
M2="${M2:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"

echo "== A. second model: dead zone replication =="
python3 -u -m kvadmission.sweep --model "$M2" --seeds 3 --length 8192 \
  --needles 8 --budgets 512 2048 --policies full recent h2o snapkv random \
  --out results/gpu-smollm2-sweep.json > results/gpu-smollm2-sweep.log 2>&1
echo "  exit=$?"

echo "== A2. second model: oracle ceiling =="
python3 -u scripts/run_oracle.py --model "$M2" --budgets 2048 --seeds 3 \
  --out results/gpu-smollm2-oracle.json > results/gpu-smollm2-oracle.log 2>&1
echo "  exit=$?"

echo "== C. distance curve, fine bins =="
python3 -u scripts/run_distance_curve.py --policies recent h2o random \
  --needles 16 --seeds 4 --bins 8 \
  --out results/gpu-distance-curve.json > results/gpu-distance-curve.log 2>&1
echo "  exit=$?"

echo "== D. query-aware admission =="
python3 -u scripts/run_queryaware.py --seeds 5 --budget 2048 \
  --out results/gpu-queryaware.json > results/gpu-queryaware.log 2>&1
echo "  exit=$?"

echo "== D2. query-aware, lower budget =="
python3 -u scripts/run_queryaware.py --seeds 5 --budget 512 \
  --out results/gpu-queryaware-512.json > results/gpu-queryaware-512.log 2>&1
echo "  exit=$?"

echo "ALL_DONE"
sleep "${DRAIN_SECONDS:-900}"
echo "== self-stopping =="
poweroff 2>/dev/null || shutdown -h now 2>/dev/null || kill -TERM 1 2>/dev/null || kill -9 1
