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

Why a subprocess (not in-process): the program files are IMPORT-SAFE (their train
block is guarded behind `if __name__ == "__main__" or AICHILLES_RUN==1`), so the
harness can import them to get `program_module.__file__` WITHOUT training. We then
launch that same file as a fresh subprocess with AICHILLES_RUN=1 to actually train.
A fresh interpreter per call is the only CUDA-safe option (the scripts hold
torch.compile + CUDA state and the harness forks its worker). PR_SET_PDEATHSIG ties
each training child's lifetime to the harness worker so a harness SIGKILL (timeout)
can't leak an orphaned GPU process.

Root-cause: on a crash we re-raise with the child's FULL traceback (its last lines
name the real file:line, e.g. best_program.py:319 `_decorr_bigram_primes[j]`
IndexError). The harness captures that in result["error"]/["traceback"], Agent 2
stores it on the matrix entry, and Agent 3 root-causes from it — no coverage needed.

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


def _app_root_for(script: str) -> str:
    """Walk up from the program file's directory to the dir that holds lib.py
    (the app root, where evaluator.py / lib.py live). The training subprocess
    gets this on PYTHONPATH so `from lib import ...` resolves regardless of whether
    the program file sits at the app root (initial_program.py) or in a best/<algo>/
    subdir (best_program.py)."""
    d = os.path.dirname(os.path.abspath(script))
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "lib.py")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(script))


def _run_once(script, base_env, seed, preexec, label):
    """Run one training subprocess at `seed`, STREAMING its live progress
    (the script's own 'step | loss | remaining' line) to stderr so the user can
    watch each training. Returns parsed metrics or raises. The harness --timeout
    is the hang backstop (it SIGKILLs the worker -> PR_SET_PDEATHSIG kills this child)."""
    env = {**base_env, "SEED": str(seed)}
    sys.stderr.write(f"      [{label} | seed {seed} | depth {env['DEPTH']} | seq {env['SEQ_LEN']} "
                     f"| bs {env['DEVICE_BATCH_SIZE']} | budget {env['TIME_BUDGET']}s]\n")
    sys.stderr.flush()
    proc = subprocess.Popen([sys.executable, script], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env, preexec_fn=preexec)
    chunks = []
    try:
        while True:
            chunk = proc.stdout.read(80)  # small chunks so the \r progress line streams live
            if chunk:
                chunks.append(chunk)
                sys.stderr.write(chunk)
                sys.stderr.flush()
            elif proc.poll() is not None:
                break
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
    sys.stderr.write("\n")
    sys.stderr.flush()
    out = "".join(chunks)
    vals = {}
    for key, rx in _METRIC_RE.items():
        m = rx.search(out)
        if m:
            vals[key] = float(m.group(1))
    if "val_bpb" not in vals:  # no result -> real crash / OOM / timeout
        raise RuntimeError(out[-1500:] or f"seed {seed}: no val_bpb (crash / OOM)")
    return vals


def run_workload(program_module, workload: dict):
    # The program file is import-safe; run IT as the training subprocess. __file__
    # is the absolute path the harness imported, so P and P' always train their own
    # real source (the same file Agent 3 reads for root-cause).
    script = program_module.__file__
    app_root = _app_root_for(script)
    base_env = {
        **os.environ,  # propagates DATA_DIR, CUDA_VISIBLE_DEVICES, AICHILLES_EAGER, etc.
        "AICHILLES_RUN": "1",  # trips the train guard when not literally __main__
        "PYTHONPATH": app_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "DEPTH":             str(workload.get("depth", 12)),
        "SEQ_LEN":           str(workload.get("seq_len", 2048)),
        "DEVICE_BATCH_SIZE": str(workload.get("device_batch_size", 64)),
        "TIME_BUDGET":       str(workload.get("time_budget", 30)),
    }
    label = os.path.basename(script)
    preexec = _pdeathsig if sys.platform.startswith("linux") else None

    base_seed = int(workload.get("seed", 42))
    n_seeds = int(workload.get("n_seeds") or os.environ.get("AICHILLES_SCREEN_SEEDS", 3))
    n_seeds = max(1, n_seeds)

    bpbs, steps, vrams, secs = [], [], [], []
    for i in range(n_seeds):  # any seed that crashes/OOMs raises -> correctness witness
        vals = _run_once(script, base_env, base_seed + i, preexec, label)
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
