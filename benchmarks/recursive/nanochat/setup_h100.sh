#!/usr/bin/env bash
# =============================================================================
# setup_h100.sh — environment for running AIChilles + recursive/nanochat
#                 leaderboard programs on a fresh H100 (Hopper) cloud GPU box.
#
# Differences vs the B200 setup:
#   • B200 (Blackwell, cap 10.0) used flash-attn-4 (cute).  H100 (Hopper, cap 9.0)
#     uses flash-attn-3, which the programs fetch via the HuggingFace `kernels`
#     package automatically. SDPA is the universal fallback if FA3 is unavailable.
#
# Usage:
#   bash setup_h100.sh            # full install into ./.venv
#   CUDA_TAG=cu121 bash setup_h100.sh   # override the torch CUDA build
#
# After it finishes:
#   source .venv/bin/activate
#   export DATA_DIR=/path/to/data           # FineWeb shards + tokenizer
#   export ANTHROPIC_API_KEY=sk-...          # for the AIChilles agents
#   python prepare.py                        # build the dataset (one-time)
# =============================================================================
set -euo pipefail

# ---- config (override via env) ----------------------------------------------
VENV="${VENV:-.venv}"
PYBIN="${PYBIN:-python3}"
# torch CUDA build. NOTE: `nvidia-smi` "CUDA Version: 13.0" is the MAX your driver
# supports — NOT a requirement. Drivers are backward-compatible, so a cu128 wheel
# runs fine on a 13.0 driver. There is no stable cu130 torch wheel; cu128 is the
# right choice (same build the B200 leaderboard used: torch 2.10+cu128).
# Fallbacks if cu128 misbehaves: cu126, then cu124.
CUDA_TAG="${CUDA_TAG:-cu128}"
# -----------------------------------------------------------------------------

say(){ printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

say "0. Sanity checks"
command -v "$PYBIN" >/dev/null || { echo "ERROR: $PYBIN not found"; exit 1; }
"$PYBIN" --version
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found — is this a GPU box with drivers installed?"
fi

# Optional system packages (skipped if no passwordless sudo / no apt).
if command -v apt-get >/dev/null && sudo -n true 2>/dev/null; then
  say "1. System packages (build tools, git)"
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends build-essential git python3-venv python3-dev
else
  echo "(skipping apt — no sudo or not Debian/Ubuntu; assuming build tools already present)"
fi

say "2. Python venv"
"$PYBIN" -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

say "3. PyTorch ($CUDA_TAG build)"
# Latest stable torch for the chosen CUDA build. H100 needs a recent torch for
# SDPA + torch.compile + FA3 ABI compatibility.
pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" torch

say "4. nanochat harness + data deps"
# lib.py / prepare.py use: tiktoken (tokenizer), pyarrow (parquet shards),
# huggingface_hub + datasets (FineWeb download), numpy.
pip install numpy tiktoken pyarrow "huggingface_hub>=0.24" datasets requests tqdm

say "5. Flash-Attention 3 (Hopper) via HuggingFace kernels"
# The programs do `from kernels import get_kernel; get_kernel('kernels-community/flash-attn3')`
# on cap (9,0). `kernels` fetches a prebuilt FA3 binary at first use (no compile).
# If this import path ever fails at runtime, the import-safe transform's SDPA
# override is the universal fallback (works on any GPU, no extra install).
pip install kernels || echo "WARNING: 'kernels' install failed — you can rely on the SDPA fallback instead."

say "6. AIChilles pipeline deps"
pip install anthropic matplotlib scipy

say "7. Verify torch + GPU"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print("device:", torch.cuda.get_device_name(0), "| capability:", cap,
          "(9,0)=H100 Hopper -> FA3 path" if cap == (9,0) else "")
    # SDPA smoke test (the universal attention fallback)
    import torch.nn.functional as F
    q = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    o = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    print("SDPA causal attention OK, out:", tuple(o.shape))
PY

say "8. Optional: fetch the FA3 kernel now (so first training doesn't pay it)"
python - <<'PY' || echo "(FA3 prefetch skipped/failed — SDPA fallback still works)"
try:
    from kernels import get_kernel
    import torch
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0)
    repo = "varunneal/flash-attention-3" if cap == (9,0) else "kernels-community/flash-attn3"
    get_kernel(repo)
    print("FA3 kernel fetched:", repo)
except Exception as e:
    raise SystemExit(f"FA3 not available: {e}")
PY

say "DONE"
cat <<'EOF'

Next steps:
  source .venv/bin/activate
  export DATA_DIR=/path/to/data          # where FineWeb shards + tokenizer live
  export ANTHROPIC_API_KEY=sk-...
  python prepare.py                       # one-time dataset build (downloads FineWeb)

  # smoke-test one program (eager + short eval keeps it fast):
  AICHILLES_EAGER=1 AICHILLES_EVAL_TOKENS=2097152 SEQ_LEN=256 DEPTH=8 \
    DEVICE_BATCH_SIZE=32 TIME_BUDGET=30 AICHILLES_RUN=1 \
    python recursive/vanilla_transformer.py

Notes:
  • H100 has 80 GB vs B200's ~178 GB. The n-gram-heavy leaderboard programs
    (ranks 1-2) may OOM — trim DEVICE_BATCH_SIZE / n-gram table size for those,
    and treat any OOM as a hardware-tier artifact, not a program weakness.
  • Two H100s: pin P and P' to separate GPUs with CUDA_VISIBLE_DEVICES=0 / =1
    to run the differential in parallel without shared-GPU contention.
EOF
