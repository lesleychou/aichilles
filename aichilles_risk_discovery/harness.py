"""
Execution harness with optional sys.settrace coverage collection.

run_one(program_path, run_workload_code, workload, timeout, collect_coverage) -> dict
  Returns {output, time, mem_bytes, coverage, error, traceback}.
  coverage: dict[int, int] mapping line_number -> execution_count for program_path.
            Empty dict when collect_coverage=False.
"""
import importlib.util
import multiprocessing
import os
import sys
import time
import traceback as _tb
import tracemalloc
from collections import defaultdict


def _load_program(program_path: str):
    prog_dir = os.path.dirname(os.path.abspath(program_path))
    if prog_dir not in sys.path:
        sys.path.insert(0, prog_dir)
    d = prog_dir
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "evaluator.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            break
        d = os.path.dirname(d)
    spec = importlib.util.spec_from_file_location("_prog", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _worker(program_path: str, run_workload_code: str, workload: dict,
            collect_coverage: bool, queue, extra_path: str | None = None):
    """Runs in a fresh subprocess."""
    abs_path = os.path.abspath(program_path)

    # Ensure the app directory is first on sys.path so app-local modules
    # (e.g. llm_sql/utils.py) are found before aichilles_risk_discovery/utils.py.
    # Also evict any same-named modules already cached in sys.modules from the
    # forked parent — Python won't re-import a cached module even if sys.path
    # changes, so we must invalidate them explicitly.
    app_dir = os.path.dirname(abs_path)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    # When P' lives in a best/algo/ subdirectory, app_dir points there rather
    # than the app root (where run_workload.py and app modules like txn_simulator
    # live).  extra_path is the true app root passed by the caller.
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    # Evict stale cached modules for any .py file in app_dir OR extra_path.
    # The fork inherits sys.modules from the parent, which may have loaded
    # same-named modules from aichilles_risk_discovery/ (e.g. utils.py, solver.py).
    # We must evict them so the subprocess re-imports from the correct app dir.
    _evict_dirs = [app_dir] + ([extra_path] if extra_path else [])
    for _dir in _evict_dirs:
        for fname in os.listdir(_dir):
            if fname.endswith(".py"):
                mod_name = fname[:-3]
                cached = sys.modules.get(mod_name)
                if cached is not None:
                    cached_file = getattr(cached, "__file__", "") or ""
                    if os.path.abspath(cached_file) != os.path.join(_dir, fname):
                        del sys.modules[mod_name]

    counts: dict[int, int] = defaultdict(int)

    if collect_coverage:
        def tracer(frame, event, arg):
            # On "call", decide per-frame whether to trace.
            # Returning None here skips all subsequent events for that frame,
            # which avoids paying settrace overhead inside scipy/numpy internals.
            if event == "call":
                if os.path.abspath(frame.f_code.co_filename) == abs_path:
                    return tracer  # trace this frame
                return None        # skip non-target frames entirely
            if event == "line":
                counts[frame.f_lineno] += 1
            return tracer
        sys.settrace(tracer)

    t0 = time.perf_counter()
    try:
        ns = {}
        exec(run_workload_code, ns)  # noqa: S102
        program = _load_program(program_path)
        tracemalloc.start()
        output = ns["run_workload"](program, workload)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.perf_counter() - t0
        if collect_coverage:
            sys.settrace(None)
        queue.put({
            "output": output, "time": elapsed, "mem_bytes": peak_bytes,
            "coverage": dict(counts), "error": None, "traceback": None,
        })
    except Exception as exc:
        tracemalloc.stop()
        if collect_coverage:
            sys.settrace(None)
        elapsed = time.perf_counter() - t0
        queue.put({
            "output": None, "time": elapsed, "mem_bytes": 0,
            "coverage": dict(counts), "error": str(exc),
            "traceback": _tb.format_exc(),
        })


def run_one(
    program_path: str,
    run_workload_code: str,
    workload: dict,
    timeout: int = 30,
    collect_coverage: bool = False,
    app_dir: str | None = None,
) -> dict:
    """
    Run program_path on workload in a fresh process.
    Returns {output, time, mem_bytes, coverage, error, traceback}.
    app_dir: optional app root to prepend to sys.path in the subprocess so that
             app-local modules (e.g. txn_simulator) can be imported when
             program_path lives in a best/algo/ subdirectory.
    """
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_worker,
        args=(program_path, run_workload_code, workload, collect_coverage, queue, app_dir),
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"output": None, "time": float(timeout), "mem_bytes": 0,
                "coverage": {}, "error": "timeout", "traceback": None}

    if queue.empty():
        return {"output": None, "time": float(timeout), "mem_bytes": 0,
                "coverage": {}, "error": "process exited without result",
                "traceback": None}

    return queue.get()
