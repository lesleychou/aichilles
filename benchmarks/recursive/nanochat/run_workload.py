"""
Fixed per-app run_workload for the recursive nanochat app.

Runs ONE workload of a program (P = initial_program, P' = best_program) across
N_SEEDS training subprocesses and returns the SEED-AVERAGED metrics for the
AIChilles oracle. Averaging is the fix for single-run noise: nanochat training is
stochastic and wall-clock-budgeted, so a single (program, workload) run is noisy
enough to produce false optimality witnesses. We compare means instead.

Seed policy (controls cost):
  n_seeds = workload["n_seeds"]  (if agent2 injected one)  else
            $AICHILLES_SCREEN_SEEDS                        else  3
  The orchestrator screens at 3; on a flagged regression agent2 re-runs the same
  workload with n_seeds = --confirm-seeds (e.g. 5) and keeps the witness only if
  it reproduces. The coverage-only run is forced to n_seeds=1 (its output is unused
  by the oracle), so a screened workload costs ~ 1 + 3 + 3 trainings.

Why a subprocess (not in-process): the recursive scripts train at module import,
hold torch.compile + CUDA state, and the harness forks its worker — a fresh
interpreter per call is the only CUDA-safe option. PR_SET_PDEATHSIG ties each
training child's lifetime to the harness worker so a harness SIGKILL (timeout)
can't leak an orphaned GPU process.

IMPORTANT: the harness timeout (run_all_app --timeout) must cover ALL n_seeds
trainings in one call, since they run serially here.

Workload params (JSON-serializable scalars; Agent 1 infers ranges from evaluator.py).
P and P' MUST receive the same workload so the differential is fair:
  seed              int   base RNG seed; seeds used are base..base+n_seeds-1  [true input]
  seq_len           int   sequence length (<= 2048)                          [true input]
  time_budget       int   per-seed training seconds (small = cheap proxy)
  depth             int   transformer layers (n_layer)      [config-robustness probe]
  device_batch_size int   per-step batch                    [config-robustness probe]
NOTE: depth/device_batch_size are program *config*, not true inputs — varying them
tests config-robustness (does P' generalize beyond the regime it was tuned for).

Returned metrics. The optimality oracle is HIGHER-IS-BETTER and only scores
int/float fields, so we expose exactly ONE numeric field for it:
  neg_val_bpb = -mean(val_bpb over seeds) -> fires only when P' mean bpb is WORSE
                                             (higher) than P -> a real regression.
Everything else is a STRING (ignored by the oracle, kept for the report): per-seed
bpb, std (noise gauge), step count, peak VRAM. Dropped as numeric oracle fields:
peak_vram_mb (P' is constantly heavier -> would fire on every workload; OOM ->
correctness instead) and num_steps (cross-architecture step counts aren't a clean
regression). A crash / CUDA-OOM / no-output run RAISES -> correctness flags it.
Success is keyed on the printed "val_bpb:" line, NOT the exit code (the scripts can
exit non-zero from a benign native-threadpool teardown abort after printing).
"""
import os
import re
import signal
import subprocess
import sys

_METRIC_RE = {
    "val_bpb":          re.compile(r"val_bpb:\s*([-\d.]+)"),
    "training_seconds": re.compile(r"training_seconds:\s*([-\d.]+)"),
    "peak_vram_mb":     re.compile(r"peak_vram_mb:\s*([-\d.]+)"),
    "num_steps":        re.compile(r"num_steps:\s*([-\d.]+)"),
}


def _pdeathsig():  # pragma: no cover - Linux child-process setup
    """If the harness worker (our parent) dies, the kernel SIGKILLs this training
    child too — prevents orphaned GPU processes on a harness timeout."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def _run_once(script, base_env, seed, backstop, preexec):
    """Run one training subprocess at `seed`; return parsed metrics or raise."""
    env = {**base_env, "SEED": str(seed)}
    try:
        p = subprocess.run([sys.executable, script], capture_output=True, text=True,
                           env=env, timeout=backstop, preexec_fn=preexec)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"training (seed {seed}) exceeded {backstop}s backstop") from exc
    out, err = p.stdout, p.stderr
    vals = {}
    for key, rx in _METRIC_RE.items():
        m = rx.search(out)
        if m:
            vals[key] = float(m.group(1))
    if "val_bpb" not in vals:  # no result -> real crash / OOM / timeout
        raise RuntimeError((err or out)[-1500:] or f"seed {seed}: no val_bpb (crash / OOM)")
    return vals


def run_workload(program_module, workload: dict):
    base_env = {
        **os.environ,  # propagates DATA_DIR, CUDA_VISIBLE_DEVICES, etc.
        "DEPTH":             str(workload.get("depth", 12)),
        "SEQ_LEN":           str(workload.get("seq_len", 2048)),
        "DEVICE_BATCH_SIZE": str(workload.get("device_batch_size", 64)),
        "TIME_BUDGET":       str(workload.get("time_budget", 30)),
    }
    script = program_module.PROGRAM_SCRIPT
    backstop = int(base_env["TIME_BUDGET"]) + 1200  # per-seed safety; harness timeout is primary
    preexec = _pdeathsig if sys.platform.startswith("linux") else None

    base_seed = int(workload.get("seed", 42))
    n_seeds = int(workload.get("n_seeds") or os.environ.get("AICHILLES_SCREEN_SEEDS", 3))
    n_seeds = max(1, n_seeds)

    bpbs, steps, vrams, secs = [], [], [], []
    for i in range(n_seeds):  # any seed that crashes/OOMs raises -> correctness witness
        vals = _run_once(script, base_env, base_seed + i, backstop, preexec)
        bpbs.append(vals["val_bpb"])
        steps.append(vals.get("num_steps", 0.0))
        vrams.append(vals.get("peak_vram_mb", 0.0))
        secs.append(vals.get("training_seconds", 0.0))

    mean_bpb = sum(bpbs) / len(bpbs)
    std_bpb = (sum((b - mean_bpb) ** 2 for b in bpbs) / len(bpbs)) ** 0.5
    return {
        # ONLY numeric optimality signal: mean over seeds, higher-is-better -> fires
        # only when P' mean bpb is worse (higher) than P.
        "neg_val_bpb": -mean_bpb,
        # Informational only (strings -> oracle ignores them).
        "n_seeds":          str(n_seeds),
        "val_bpb_mean":     f"{mean_bpb:.6f}",
        "val_bpb_std":      f"{std_bpb:.6f}",       # noise gauge: trust a witness only if gap >> std
        "val_bpb_seeds":    ",".join(f"{b:.6f}" for b in bpbs),
        "num_steps":        str(int(sum(steps) / len(steps))),
        "peak_vram_mb":     f"{max(vrams):.1f}",
        "training_seconds": f"{sum(secs):.1f}",
    }
