#!/usr/bin/env bash
# Run a phase and STOP THE BOX when finished.
#
# Why this exists: a previous run finished in ~2h and then idled for ~16h,
# burning $2.63. With vast.ai auto-top-up enabled there is no natural brake -
# an idle instance bills indefinitely against your card. The container
# poweroff below moves the instance to 'exited', which ends GPU billing
# without needing an API key on this (third-party) machine.
#
# Usage on the box:  setsid nohup bash scripts/run_selfstop.sh > /tmp/run.log 2>&1 < /dev/null &
set -uo pipefail

cd "$(dirname "$0")/.."
BUDGET="${BUDGET:-2048}"
LENGTH="${LENGTH:-8192}"
STEPS="${STEPS:-300}"
BETA="${BETA:-1.0}"
MIX="${MIX:-0.7}"
TAG="${TAG:-2c-b1}"

mkdir -p results

echo "== train ($TAG) =="
python3 -u -m kvadmission.train \
  --steps "$STEPS" --budget "$BUDGET" --length "$LENGTH" \
  --beta "$BETA" --mix "$MIX" --entropy-coef 0.01 \
  --out "results/policy-$TAG" > "results/train-$TAG.log" 2>&1
echo "train exit=$?"

echo "== eval ($TAG) =="
python3 -u -m kvadmission.evaluate \
  --checkpoint "results/policy-$TAG/policy-best.pt" \
  --budget "$BUDGET" --seeds 5 \
  --out "results/phase-$TAG-eval.json" > "results/phase-$TAG-eval.log" 2>&1
echo "eval exit=$?"

echo "ALL_DONE"

# Give a watcher a moment to notice ALL_DONE and rsync results out.
sleep "${DRAIN_SECONDS:-300}"

# End GPU billing. Try the graceful paths first; kill PID 1 as the fallback,
# which terminates the container on images without a real init.
echo "== self-stopping =="
poweroff 2>/dev/null || shutdown -h now 2>/dev/null || kill -TERM 1 2>/dev/null || kill -9 1
