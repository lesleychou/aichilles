# aichilles_risk_discovery/run_all_app.py
"""
Orchestrator: chain Agent 1 → Agent 2 (x4) → Agent 3 for one ADRS app.

Usage:
  python run_all_app.py --app eplb \
    --best_program benchmarks/ADRS/eplb/best/claude/adaevolve/best_program.py \
    [--budget 200] [--patience 5] [--theta 0.1] \
    [--skip_agent1] [--skip_agent2] [--results_dir path/to/existing/run] \
    [--agent2_types scalab_time scalab_mem optimality]

--agent2_types selects which of the 4 type-agents to run (default: all four).
--skip_agent2 is shorthand for running none of them.
Both flags are compatible with --results_dir to resume a partial run.
"""
import argparse
import datetime
import json
from pathlib import Path

_BENCH_ROOT = Path(__file__).parent.parent / "benchmarks"
_ADRS_ROOT = _BENCH_ROOT / "ADRS"
# App roots searched in order. Apps may live under benchmarks/ADRS/ or
# benchmarks/recursive/ (e.g. the recursive nanochat app).
_APP_ROOTS = [_ADRS_ROOT, _BENCH_ROOT / "recursive"]
_RESULTS_ROOT = Path(__file__).parent / "results"

APPS = ["cloudcast", "eplb", "llm_sql", "prism", "txn_scheduling", "nanochat"]


def _resolve_app_dir(app: str) -> Path:
    for root in _APP_ROOTS:
        candidate = root / app
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"App directory not found for '{app}' under: "
        + ", ".join(str(r) for r in _APP_ROOTS)
    )


def _merge_matrix_v(results_dir: Path) -> None:
    """Merge 4 per-agent matrix_V_{sig}.json files into matrix_V.json for Agent 3."""
    sigs = ["correctness", "scalab_time", "scalab_mem", "optimality"]
    combined = []
    for sig in sigs:
        path = results_dir / f"matrix_V_{sig}.json"
        if path.exists():
            try:
                combined.extend(json.loads(path.read_text()))
            except Exception as exc:
                print(f"[orchestrator] WARNING: failed to parse {path}: {exc}", file=__import__('sys').stderr)
    (results_dir / "matrix_V.json").write_text(json.dumps(combined, indent=2, default=str))
    print(f"[orchestrator] merged {len(combined)} matrix entries → matrix_V.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True, choices=APPS)
    parser.add_argument("--best_program", required=True)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=30,
                        help="Per-call harness timeout (s). Raise for slow apps "
                             "like nanochat whose programs train a model per call. "
                             "Must cover ALL screen-seeds trainings in one call.")
    parser.add_argument("--screen-seeds", type=int, default=3,
                        help="Seeds averaged per oracle call to cut single-run noise "
                             "(nanochat run_workload reads $AICHILLES_SCREEN_SEEDS).")
    parser.add_argument("--confirm-seeds", type=int, default=0,
                        help="On a flagged regression, re-run the workload at this many "
                             "seeds and keep the witness only if it reproduces. 0 = off.")
    parser.add_argument("--eager", action="store_true",
                        help="Run nanochat programs WITHOUT torch.compile (sets "
                             "$AICHILLES_EAGER=1). Much faster per training during search; "
                             "differential stays fair (both P and P' run eager).")
    parser.add_argument("--eval-tokens", type=int, default=0,
                        help="Override nanochat eval length ($AICHILLES_EVAL_TOKENS), e.g. "
                             "2097152 (~10x shorter eval, saves ~15s/training). 0 = script default.")
    _SIG_NAMES = ["correctness", "scalab_time", "scalab_mem", "optimality"]
    parser.add_argument("--skip_agent1", action="store_true",
                        help="Skip Agent 1 — use existing grammar.json in results_dir")
    parser.add_argument("--skip_agent2", action="store_true",
                        help="Skip all Agent 2 types (shorthand for --agent2_types with no args)")
    parser.add_argument("--agent2_types", nargs="*", choices=_SIG_NAMES, default=None,
                        metavar="SIG",
                        help="Agent 2 types to run, e.g. --agent2_types scalab_time optimality "
                             "(default: all four; overridden to empty by --skip_agent2)")
    parser.add_argument("--results_dir", default=None,
                        help="Reuse an existing results directory (for --skip_agent1/2)")
    args = parser.parse_args()

    # Screen-seed count and eager mode are read by the training subprocess via env
    # (inherited through the forked workers + run_workload).
    import os
    os.environ["AICHILLES_SCREEN_SEEDS"] = str(args.screen_seeds)
    if args.eager:
        os.environ["AICHILLES_EAGER"] = "1"
    if args.eval_tokens:
        os.environ["AICHILLES_EVAL_TOKENS"] = str(args.eval_tokens)
    # nanochat trains in a subprocess, so the harness's in-process sys.settrace
    # coverage only sees the program's import-time def lines (identical for every
    # workload) — useless for novelty AND for Agent 3's trigger/anomalous-lines.
    # Disable it: Agent 2 falls back to param-space novelty, and Agent 3 root-causes
    # from the crash traceback (run_workload surfaces the real file:line).
    if args.app == "nanochat":
        os.environ["AICHILLES_NO_COVERAGE"] = "1"

    # Lazy imports to avoid top-level import failures when running --help
    import anthropic
    from agent1_infer import run_agent1
    from agent2_explore import run_agent_type
    from agent3_analyze import run_agent3
    from knowledge_base import load_knowledge_base, save_knowledge_base, extract_transferable
    from oracle import Signature

    app_dir = _resolve_app_dir(args.app)
    best_program_path = Path(args.best_program).resolve()

    if not app_dir.exists():
        raise SystemExit(f"App directory not found: {app_dir}")
    if not best_program_path.exists():
        raise SystemExit(f"best_program not found: {best_program_path}")

    # Create timestamped results directory nested under the app folder
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = _RESULTS_ROOT / args.app / f"{args.app}_{ts}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[orchestrator] results_dir: {results_dir}")

    # Resolve which agent2 types to run
    _SIG_NAMES = ["correctness", "scalab_time", "scalab_mem", "optimality"]
    if args.skip_agent2:
        selected_sig_names: list[str] = []
    elif args.agent2_types is not None:
        selected_sig_names = args.agent2_types
    else:
        selected_sig_names = _SIG_NAMES

    # Save run config for reproducibility
    config = {
        "app": args.app,
        "best_program": str(best_program_path),
        "budget": args.budget,
        "patience": args.patience,
        "theta": args.theta,
        "skip_agent1": args.skip_agent1,
        "agent2_types": selected_sig_names,
        "timestamp": ts,
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2))

    client = anthropic.Anthropic()

    # Agent 1
    if not args.skip_agent1:
        run_agent1(app_dir, best_program_path, results_dir, client)
    else:
        print("[orchestrator] skipping Agent 1")

    # Agent 2 — type-specialized agents (subset or all four)
    sig_map = {
        "correctness": Signature.CORRECTNESS,
        "scalab_time": Signature.SCALAB_TIME,
        "scalab_mem":  Signature.SCALAB_MEM,
        "optimality":  Signature.OPTIMALITY,
    }
    selected_sigs = [sig_map[s] for s in selected_sig_names]

    if selected_sigs:
        n_agents = len(selected_sigs)
        budget_per_agent = args.budget // n_agents
        if args.budget % n_agents != 0:
            print(f"[orchestrator] WARNING: budget {args.budget} not evenly divisible by "
                  f"{n_agents} agent(s); effective budget={budget_per_agent * n_agents}")
        crash_workloads: list[dict] = []

        for sig in selected_sigs:
            print(f"\n[orchestrator] running Agent 2 — {sig.value} "
                  f"(budget={budget_per_agent})")
            _, new_crashes = run_agent_type(
                sig=sig,
                app_dir=app_dir,
                best_program_path=best_program_path,
                results_dir=results_dir,
                client=client,
                budget=budget_per_agent,
                patience=args.patience,
                theta=args.theta,
                timeout=args.timeout,
                confirm_seeds=args.confirm_seeds,
                crash_workloads=crash_workloads,
            )
            crash_workloads = crash_workloads + new_crashes
            if sig == Signature.CORRECTNESS:
                print(f"[orchestrator] {len(new_crashes)} crash workloads collected from correctness agent")
    else:
        print("[orchestrator] skipping Agent 2 — no types selected")

    # Always merge to produce matrix_V.json (empty [] when no agents ran, so Agent 3 can proceed)
    _merge_matrix_v(results_dir)

    # Agent 3
    run_agent3(app_dir, best_program_path, results_dir, client)

    # Update knowledge base
    matrix_path = results_dir / "matrix_V.json"
    if matrix_path.exists():
        entries = json.loads(matrix_path.read_text())
        kb_dir = _RESULTS_ROOT / args.app
        existing_kb = load_knowledge_base(kb_dir)
        new_kb = extract_transferable(entries, existing_kb if existing_kb["bug_seeds"] else None)
        save_knowledge_base(_RESULTS_ROOT, args.app, new_kb)
        print(f"[orchestrator] knowledge base updated at {kb_dir / 'knowledge_base.json'}")

    print(f"[orchestrator] done. Results at {results_dir}")


if __name__ == "__main__":
    main()
