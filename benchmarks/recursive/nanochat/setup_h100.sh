#!/usr/bin/env bash
# =============================================================================
# setup_h100.sh — environment for running AIChilles + recursive/nanochat
#                 leaderboard programs on a fresh H100 (Hopper) cloud GPU box.
#
# H100 uses flash-attn-3 (via the HuggingFace `kernels` package), not flash-attn-4.
# SDPA is the universal fallback if FA3 is unavailable.
#
# Robustness: every package is installed into ./.venv via that venv's OWN python
# (never the system python), torch auto-falls-back across CUDA tags, and the script
# HARD-VERIFIES the critical imports at the end so it fails loudly, not silently.
#
# Usage:
#   bash setup_h100.sh                  # full install into ./.venv
#   CUDA_TAG=cu126 bash setup_h100.sh   # force a specific torch CUDA build first
# =============================================================================
set -euo pipefail

# Re-exec under bash if invoked via sh/dash (source + pipefail need bash).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# ---- config (override via env) ----------------------------------------------
VENV="${VENV:-.venv}"
PYBIN="${PYBIN:-python3}"
# `nvidia-smi` "CUDA Version: 13.0" is the MAX your driver supports, not a
# requirement — a cu128 wheel runs fine on it. We try these tags in order until
# one has a wheel for your Python; cu130 isn't a stable torch build.
CUDA_TAGS=("${CUDA_TAG:-cu128}" cu126 cu124 cu121)
# -----------------------------------------------------------------------------

say(){ printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die(){ printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# All package installs go through the venv's python explicitly.
VPY=""                                  # set after venv creation
vpip(){ "$VPY" -m pip "$@"; }

say "0. Sanity checks"
command -v "$PYBIN" >/dev/null || die "$PYBIN not found"
"$PYBIN" --version
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found — is this a GPU box with drivers installed?"
fi

# Optional system packages (only with passwordless sudo + apt).
if command -v apt-get >/dev/null && sudo -n true 2>/dev/null; then
  say "1. System packages (build tools, git)"
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends build-essential git python3-venv python3-dev
else
  echo "(skipping apt — no sudo or not Debian/Ubuntu; assuming build tools present)"
fi

say "2. Python venv"
"$PYBIN" -m venv "$VENV" || die "venv creation failed"
VPY="$(cd "$(dirname "$VENV")" && pwd)/$(basename "$VENV")/bin/python"
[ -x "$VPY" ] || die "venv python not found at $VPY"
echo "venv python: $VPY"
"$VPY" --version
vpip install --upgrade pip wheel setuptools

say "3. PyTorch (auto-fallback across CUDA tags)"
TORCH_OK=0; TORCH_TAG=""
declare -A _seen=()
for tag in "${CUDA_TAGS[@]}"; do
  [ -n "${_seen[$tag]:-}" ] && continue; _seen[$tag]=1
  echo "--- trying torch from $tag ..."
  if vpip install --index-url "https://download.pytorch.org/whl/$tag" torch; then
    TORCH_OK=1; TORCH_TAG="$tag"; break
  fi
  echo "    ($tag failed — trying next)"
done
[ "$TORCH_OK" = 1 ] || die "could not install torch for $("$VPY" -V) from any CUDA tag (${CUDA_TAGS[*]}). Check Python version (3.10–3.12 best) and network."
echo "torch installed from: $TORCH_TAG"

say "4. nanochat harness + data deps"
vpip install numpy tiktoken pyarrow "huggingface_hub>=0.24" datasets requests tqdm

say "5. Flash-Attention 3 (Hopper) via HuggingFace kernels"
# Programs do `get_kernel('kernels-community/flash-attn3')` on cap (9,0). Prebuilt,
# no compile. Non-fatal: the SDPA fallback works on any GPU with no extra install.
vpip install kernels || echo "WARNING: 'kernels' install failed — relying on SDPA fallback is fine."

say "6. AIChilles pipeline deps"
vpip install anthropic matplotlib scipy

say "7. HARD verify (fails loudly if anything is missing)"
"$VPY" - <<'PY' || die "verification failed — the venv is NOT ready (see traceback above)"
import importlib.util, sys
required = ["torch", "numpy", "tiktoken", "pyarrow", "huggingface_hub", "anthropic", "matplotlib"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"MISSING modules in venv: {missing}")
import torch
print("python :", sys.executable)
print("torch  :", torch.__version__, "| cuda build:", torch.version.cuda)
print("cuda   :", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print("device :", torch.cuda.get_device_name(0), "| capability:", cap,
          "(9,0)=H100 Hopper -> FA3 path" if cap == (9, 0) else "")
    import torch.nn.functional as F
    q = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    F.scaled_dot_product_attention(q, q, q, is_causal=True)
    print("SDPA causal attention: OK")
else:
    print("WARNING: CUDA not visible to torch — check drivers / CUDA_VISIBLE_DEVICES")
print("ALL REQUIRED MODULES PRESENT IN VENV ✓")
PY

say "8. (optional) prefetch the FA3 kernel"
"$VPY" - <<'PY' || echo "(FA3 prefetch skipped — SDPA fallback still works)"
from kernels import get_kernel
import torch
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0)
repo = "varunneal/flash-attention-3" if cap == (9,0) else "kernels-community/flash-attn3"
get_kernel(repo); print("FA3 kernel fetched:", repo)
PY

say "DONE — venv ready: $VENV"
cat <<EOF

Activate and use THIS venv (the one above):
  source $VENV/bin/activate
  python -c "import torch; print(torch.__version__)"   # sanity: should NOT error

Then:
  export DATA_DIR=\$HOME/nanochat_data && mkdir -p "\$DATA_DIR"
  export ANTHROPIC_API_KEY=sk-...
  python prepare.py
EOF
