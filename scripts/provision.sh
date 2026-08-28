#!/usr/bin/env bash
# Provision a rented GPU box for phase 2 training.
# Usage (on the remote box):  bash provision.sh
#
# Assumes: Ubuntu + CUDA image, python3.10+, network access to HuggingFace.
# Everything here is idempotent - safe to re-run after a dropped connection.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/kvAdmission}"
STEPS="${STEPS:-300}"
BUDGET="${BUDGET:-2048}"
LENGTH="${LENGTH:-8192}"

echo "== installing deps =="
pip install -q --upgrade pip

# Every one of these was a real failure on a stock pytorch/pytorch CUDA image:
#  - transformers 5.x needs torch >= 2.5; on 2.4 it silently disables the torch
#    backend and every model class becomes un-importable
#  - torchvision/torchaudio are built against the image's original torch, so
#    upgrading torch breaks their ABI ("operator torchvision::nms does not
#    exist") and transformers explodes on import. We are text-only; remove them.
#  - triton JIT-compiles kernels and needs a C compiler the image lacks
#  - conda's readline can segfault pytest at startup (_readline_workaround)
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
pip install -q "torch>=2.5" "transformers>=5.0" pytest
command -v gcc >/dev/null || {
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc g++
}
python3 -c "import readline" 2>/dev/null || conda install -y -q readline

echo "== sanity: GPU visible? =="
python3 -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device - wrong image or wrong box'
p = torch.cuda.get_device_properties(0)
print(f'{p.name}, {p.total_memory/1e9:.0f} GB')
"

cd "$REPO_DIR"

echo "== pulling the base model =="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-1.7B', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
print('model ready')
"

echo "== test suite (fails here means do not bother training) =="
python3 -m pytest tests/ -q

echo "== training: $STEPS steps, B=$BUDGET, len=$LENGTH =="
python3 -u -m kvadmission.train \
  --steps "$STEPS" --budget "$BUDGET" --length "$LENGTH" \
  --out results/policy 2>&1 | tee results/train.log

echo "== evaluating against the phase 1 needle harness =="
python3 -u -m kvadmission.evaluate \
  --checkpoint results/policy/policy-best.pt \
  --budget "$BUDGET" --seeds 5 2>&1 | tee results/phase2-eval.log

echo
echo "== done. pull these back: =="
echo "  results/policy/policy-best.pt trained head (best held-out KL)"
echo "  results/policy/history.json   training metrics"
echo "  results/phase2-eval.json      eval rows in phase 1 schema"
echo "  results/*.log"
