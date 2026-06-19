# aichilles_risk_discovery/run_parallel.py
"""
Run risk discovery for all 5 apps x 5 programs in parallel.

Usage:
  python run_parallel.py [--budget 200] [--patience 5] [--theta 0.1]
                         [--workers 5] [--apps app1 app2 ...]
                         [--programs model/algo ...]
                         [--agent2_types correctness scalab_time scalab_mem optimality]
                         [--dry_run]

--workers controls parallel apps (default: 5 — one per app). Programs within each app
always run sequentially so each one warm-starts from the previous program's KB findings.
--dry_run prints the commands that would run without executing them.

Logs for each job are written to results/<app>/<job_name>.log.
A summary table is printed when all jobs finish.
"""
import argparse
import datetime
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_TAIL_LINES = 30  # log lines to print on failure

_HERE      = Path(__file__).parent
_ADRS_ROOT = _HERE.parent / "benchmarks" / "ADRS"
_RESULTS   = _HERE / "new_results"

ALL_APPS = ["cloudcast", "eplb", "llm_sql", "prism", "txn_scheduling"]
ALL_PROGRAMS = [
    "claude/adaevolve",
    "claude/engram",
    "claude/openevolve",
    "gpt/adaevolve",
    "gpt/engram",
    "gpt/openevolve",
]


def _build_job(app: str, program: str, args: argparse.Namespace, ts: str) -> dict:
    best_program = _ADRS_ROOT / app / "best" / program / "best_program.py"
    label = f"{app}__{program.replace('/', '_')}_{ts}"
    results_dir = _RESULTS / app / label
    log_path = results_dir / "run.log"
    cmd = [
        sys.executable, str(_HERE / "run_all_v2.py"),
        "--app", app,
        "--best_program", str(best_program),
        "--budget", str(args.budget),
        "--patience", str(args.patience),
        "--theta", str(args.theta),
        "--results_dir", str(results_dir),
    ]
    if args.agent2_types:
        cmd += ["--agent2_types"] + args.agent2_types
    return {"app": app, "program": program, "label": label,
            "cmd": cmd, "log_path": log_path, "results_dir": results_dir,
            "best_program": best_program}


def _run_job(job: dict) -> dict:
    job["log_path"].parent.mkdir(parents=True, exist_ok=True)
    start = datetime.datetime.now()
    with open(job["log_path"], "w") as log:
        log.write(f"# {job['label']}\n# started: {start.isoformat()}\n# cmd: {' '.join(job['cmd'])}\n\n")
        log.flush()
        proc = subprocess.run(
            job["cmd"],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(_HERE),
        )
    elapsed = (datetime.datetime.now() - start).total_seconds()
    return {**job, "returncode": proc.returncode, "elapsed": elapsed}


def _run_app_group(app: str, jobs: list[dict], print_fn) -> list[dict]:
    """Run all programs for one app sequentially so KB knowledge accumulates."""
    results = []
    for i, job in enumerate(jobs):
        print_fn(f"  [app={app} {i+1}/{len(jobs)}] starting {job['label']}")
        print_fn(f"    log: {job['log_path']}")
        r = _run_job(job)
        status = "OK " if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
        print_fn(f"  [app={app} {i+1}/{len(jobs)}] [{status}] {r['label']}  ({r['elapsed']:.0f}s)")
        if r["returncode"] != 0:
            log_path: Path = r["log_path"]
            if log_path.exists():
                lines = log_path.read_text().splitlines()
                tail = lines[-_TAIL_LINES:]
                print_fn(f"    --- last {len(tail)} lines of {log_path} ---")
                for line in tail:
                    print_fn(f"    {line}")
                print_fn(f"    --- end of log ---")
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget",   type=int,   default=200)
    parser.add_argument("--patience", type=int,   default=5)
    parser.add_argument("--theta",    type=float, default=0.1)
    parser.add_argument("--workers",  type=int,   default=5,
                        help="Max parallel jobs (default: 5)")
    parser.add_argument("--apps",     nargs="+",  default=ALL_APPS,
                        choices=ALL_APPS, metavar="APP")
    parser.add_argument("--programs", nargs="+",  default=ALL_PROGRAMS,
                        metavar="MODEL/ALGO",
                        help="Programs to run, e.g. claude/adaevolve gpt/openevolve")
    _SIG_NAMES = ["correctness", "scalab_time", "scalab_mem", "optimality"]
    parser.add_argument("--agent2_types", nargs="*", choices=_SIG_NAMES,
                        default=None, metavar="SIG")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs = []
    missing = []
    for app in args.apps:
        for program in args.programs:
            job = _build_job(app, program, args, ts)
            if not job["best_program"].exists():
                missing.append(str(job["best_program"]))
            else:
                jobs.append(job)

    if missing:
        print(f"WARNING: skipping {len(missing)} missing best_program paths:")
        for p in missing:
            print(f"  {p}")

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Scheduling {len(jobs)} jobs "
          f"({len(args.apps)} apps × {len(args.programs)} programs), "
          f"workers={args.workers}, budget={args.budget}\n")

    if args.dry_run:
        for job in jobs:
            print(f"  [{job['label']}]")
            print(f"    cmd: {' '.join(job['cmd'])}")
            print(f"    dir: {job['results_dir']}")
        return

    # Group jobs by app — programs within an app run sequentially to preserve KB sharing
    from collections import defaultdict
    app_groups: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        app_groups[job["app"]].append(job)

    print("To follow an app live:  tail -f new_results/<app>/*/run.log")
    print("To watch all at once:   tail -f new_results/*/*/run.log\n")
    print(f"Apps: {list(app_groups.keys())}")
    print(f"Programs per app (sequential): {[j['program'] for j in next(iter(app_groups.values()))]}\n")

    # Thread-safe print to avoid interleaved output across app threads
    import threading
    _print_lock = threading.Lock()
    def tprint(msg: str) -> None:
        with _print_lock:
            print(msg, flush=True)

    start_all = datetime.datetime.now()
    all_results: list[dict] = []
    futures = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for app, group in app_groups.items():
            f = pool.submit(_run_app_group, app, group, tprint)
            futures[f] = app
            tprint(f"  [queued] app={app}  ({len(group)} programs, sequential)")

        done_apps = 0
        tprint(f"\n{len(app_groups)} apps queued across {args.workers} workers. Waiting...\n")

        for f in as_completed(futures):
            app_results = f.result()
            all_results.extend(app_results)
            done_apps += 1
            app = futures[f]
            ok_n  = sum(1 for r in app_results if r["returncode"] == 0)
            tprint(f"  [app={app} done] {ok_n}/{len(app_results)} OK  "
                   f"({done_apps}/{len(app_groups)} apps done)")

    total_elapsed = (datetime.datetime.now() - start_all).total_seconds()

    # Summary table
    ok   = [r for r in all_results if r["returncode"] == 0]
    fail = [r for r in all_results if r["returncode"] != 0]
    print(f"\n{'='*60}")
    print(f"DONE  {len(ok)}/{len(all_results)} succeeded  |  total wall time: {total_elapsed:.0f}s")
    if fail:
        print(f"\nFailed jobs ({len(fail)}):")
        for r in fail:
            print(f"  {r['label']}  rc={r['returncode']}  log: {r['log_path']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
