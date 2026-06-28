"""
Evaluator spec for the recursive nanochat app (read by Agent 1 to infer the
workload grammar). DESCRIPTIVE only — AIChilles runs programs via run_workload.py,
which launches the training script as a subprocess. Grammar names/ranges below
must match run_workload.py's `workload.get(...)` keys.

Differential setup:
  P  = initial_program.py             -> recursive/vanilla_transformer.py
  P' = best/recursive/best_program.py -> recursive/optimized_from_karpathy.py
Each is trained for `time_budget` seconds on a fixed FineWeb-style dataset and
scored by validation bits-per-byte (lower is better). The SAME workload is run
for both; the oracle flags workloads where P' regresses.

WHAT "WORKLOAD" MEANS HERE (important framing):
  These programs are whole training scripts, so some knobs are program *config*,
  not data *inputs*. The study is therefore a CONFIG-ROBUSTNESS probe: does the
  AI-evolved P' still hold up when moved off the single regime it was tuned for
  (depth 8, B200, 5-min budget)?
    - TRUE INPUTS (closest to ADRS "workload"): seed, seq_len.
    - CONFIG-GENERALIZATION knobs: depth, device_batch_size. Sweeping these is the
      point — they expose P''s baked-in assumptions (e.g. depth-coupled per-layer
      tables) — but a quality (bpb) gap at an off-config point is a config-
      robustness regression, not an input-workload regression.

Regression types AIChilles should find:
  - correctness : P' crashes/times-out where P succeeds (e.g. depth-coupled index
                  errors at large depth).
  - optimality  : P' produces WORSE val_bpb than P on some config (neg_val_bpb).
  (scalab_time / scalab_mem are NOT used for this app: training is wall-clock
   budgeted so total time is ~constant, and harness tracemalloc can't see GPU
   memory — those regressions surface as fewer steps / OOM -> correctness.)
"""
from typing import TypedDict


# --- Workload space ---------------------------------------------------------
# Scalar knobs run_workload.py reads and exports to the training subprocess.
GRAMMAR_WORKLOAD = {
    # RNG seed (true input). P and P' share it for a fair differential.
    "seed":              {"type": "int", "min": 0, "max": 100000},
    # sequence length (true input); must be <= 2048 (rotary precompute ceiling).
    "seq_len":           {"type": "int", "min": 256, "max": 2048, "multiple_of": 128},
    # training wall-clock budget; kept SMALL so the search stays cheap (each oracle
    # call runs several trainings). Raise the max for a faithful confirmation pass.
    "time_budget":       {"type": "int", "min": 20, "max": 60},
    # transformer depth (config). Sweeping up probes depth-coupled assumptions in
    # P' (hardcoded per-layer tables -> IndexError crash at large depth).
    "depth":             {"type": "int", "min": 4, "max": 16},
    # per-step batch (config).
    "device_batch_size": {"type": "int", "min": 8, "max": 80},
}

# No separate config-space; everything is in the workload.
GRAMMAR_CONFIG: dict = {}

CONSTRAINTS = [
    "seq_len <= 2048 and should be a multiple of 128",
    "depth >= 1",
    "P and P' must receive identical workloads for a valid differential",
]


# --- Metrics returned by run_workload ---------------------------------------
# Only `neg_val_bpb` is numeric, so it is the sole field the higher-is-better
# optimality oracle scores. The rest are strings (reported, not scored).
class EvaluationResult(TypedDict, total=False):
    neg_val_bpb: float       # -val_bpb; fires only when P' bpb is worse than P
    val_bpb: str             # informational
    num_steps: str           # informational
    peak_vram_mb: str        # informational
    training_seconds: str    # informational
