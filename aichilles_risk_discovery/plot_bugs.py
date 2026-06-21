# aichilles_risk_discovery/plot_bugs.py
"""
Post-analysis: reproduce bugs and plot raw P vs P' metrics from pipeline results.

Usage:
  python plot_bugs.py <results_dir> [<results_dir> ...] \
    [--x_param <param>] [--sig <scalab_time|scalab_mem|optimality|correctness>] \
    [--max_witnesses <N>] [--out <path.png>] [--no_reproduce]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ADRS_ROOT = Path(__file__).parent.parent / "benchmarks" / "ADRS"
_SIG_ORDER = ["scalab_time", "scalab_mem", "optimality", "correctness"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_run(results_dir: Path) -> dict | None:
    """Load config.json and clusters.json from a results directory.
    Returns None (with a warning) if either file is missing."""
    config_path  = results_dir / "config.json"
    clusters_path = results_dir / "clusters.json"
    if not config_path.exists():
        print(f"WARNING: {config_path} not found, skipping", file=sys.stderr)
        return None
    if not clusters_path.exists():
        print(f"WARNING: {clusters_path} not found, skipping", file=sys.stderr)
        return None
    config   = json.loads(config_path.read_text())
    clusters = json.loads(clusters_path.read_text())
    # Load grammar workload params as x-axis candidates (numeric params only)
    grammar_path = results_dir / "grammar.json"
    x_params: list[str] = []
    if grammar_path.exists():
        try:
            grammar = json.loads(grammar_path.read_text())
            x_params = [p["name"] for p in grammar.get("grammar_workload", [])
                        if p.get("type") in ("int", "float")]
        except Exception:
            pass
    try:
        return {
            "results_dir":  results_dir,
            "run_name":     results_dir.name,
            "app":          config["app"],
            "best_program": Path(config["best_program"]),
            "config":       config,
            "clusters":     clusters,
            "x_params":     x_params,
        }
    except (KeyError, TypeError) as exc:
        print(f"WARNING: {config_path} is malformed ({exc}), skipping", file=sys.stderr)
        return None


def sample_witnesses(witnesses: list[dict], max_n: int,
                     x_param: str | None) -> list[dict]:
    """Return up to max_n witnesses, sampled evenly across the x-axis range.
    Sorting by x_param before sampling prevents clustering at one end."""
    if max_n <= 0:
        return []
    if len(witnesses) <= max_n:
        return list(witnesses)
    if x_param:
        sorted_ws = sorted(witnesses, key=lambda w: w["w"].get(x_param, 0))
    else:
        sorted_ws = list(witnesses)
    step = len(sorted_ws) / max_n
    return [sorted_ws[int(i * step)] for i in range(max_n)]


# ── Y-metric selection ────────────────────────────────────────────────────────

def pick_y_metric(signatures: list[str], rep_rp: dict,
                  rep_rpp: dict) -> tuple[str, str]:
    """Return (field_name, unit) for the y-axis from cluster signatures.
    Priority order: scalab_time > scalab_mem > optimality > correctness.
    For optimality, picks the output field with the largest relative gap."""
    for sig in _SIG_ORDER:
        if sig not in signatures:
            continue
        if sig == "scalab_time":
            return "time", "seconds"
        if sig == "scalab_mem":
            return "memory_mb", "MB"
        if sig == "optimality":
            p_out  = rep_rp.get("output")  or {}
            pp_out = rep_rpp.get("output") or {}
            best_field, best_gap = None, -1.0
            for field, pv in p_out.items():
                if not isinstance(pv, (int, float)):
                    continue
                ppv = pp_out.get(field)
                if not isinstance(ppv, (int, float)):
                    continue
                gap = abs(float(pv) - float(ppv)) / (
                    max(abs(float(pv)), abs(float(ppv))) + 1e-9)
                if gap > best_gap:
                    best_gap, best_field = gap, field
            if best_field:
                return best_field, best_field
            return "time", "seconds"
        if sig == "correctness":
            # Correctness bugs (crash/timeout) have no numeric y-metric; default to time.
            return "time", "seconds"
    return "time", "seconds"


def _y_metrics_for_cluster(cluster: dict, rep_rp: dict | None,
                           rep_rpp: dict | None) -> list[tuple[str, str, str]]:
    """Return [(sig, y_field, y_unit)] for each plottable signature in the cluster.
    Correctness is excluded — P' crashed so there is no numeric metric to compare."""
    result = []
    for sig in _SIG_ORDER:
        if sig not in cluster["signatures"] or sig == "correctness":
            continue
        if sig == "scalab_time":
            result.append((sig, "time", "seconds"))
        elif sig == "scalab_mem":
            result.append((sig, "memory_mb", "MB"))
        elif sig == "optimality":
            y_field, y_unit = "time", "seconds"  # fallback
            if rep_rp and rep_rpp:
                p_out  = rep_rp.get("output")  or {}
                pp_out = rep_rpp.get("output") or {}
                best_field, best_gap = None, -1.0
                for field, pv in p_out.items():
                    if not isinstance(pv, (int, float)):
                        continue
                    ppv = pp_out.get(field)
                    if not isinstance(ppv, (int, float)):
                        continue
                    gap = abs(float(pv) - float(ppv)) / (
                        max(abs(float(pv)), abs(float(ppv))) + 1e-9)
                    if gap > best_gap:
                        best_gap, best_field = gap, field
                if best_field:
                    y_field, y_unit = best_field, best_field
            result.append((sig, y_field, y_unit))
    return result


def extract_y_value(result: dict, field: str) -> float | None:
    """Extract the y-axis raw value from a run_one result dict.
    Returns None if unavailable."""
    if field == "time":
        return result.get("time")
    if field == "memory_mb":
        mb = result.get("mem_bytes")
        return mb / (1024.0 * 1024.0) if mb is not None else None
    out = result.get("output") or {}
    val = out.get(field)
    return float(val) if isinstance(val, (int, float)) else None


# ── Harness re-runs ───────────────────────────────────────────────────────────

def rerun_pair(initial_path: Path, best_path: Path, run_workload_code: str,
               workload: dict, app_dir: Path) -> tuple[dict, dict]:
    """Re-run P (initial) and P' (best) on workload. Returns (rp, rpp).
    Uses harness.run_one for subprocess isolation (same as pipeline)."""
    from harness import run_one
    rp  = run_one(str(initial_path), run_workload_code, workload,
                  timeout=60, collect_coverage=False, app_dir=str(app_dir))
    rpp = run_one(str(best_path),    run_workload_code, workload,
                  timeout=60, collect_coverage=False, app_dir=str(app_dir))
    return rp, rpp


def rerun_witnesses(sampled: list[dict], initial_path: Path, best_path: Path,
                    run_workload_code: str,
                    app_dir: Path) -> list[tuple[dict, dict, dict]]:
    """Re-run P and P' for each sampled witness.
    Returns list of (witness_entry, rp_result, rpp_result).
    Silently drops any witness where either program errors."""
    results = []
    for w_entry in sampled:
        rp, rpp = rerun_pair(initial_path, best_path, run_workload_code,
                             w_entry["w"], app_dir)
        if rp.get("error") or rpp.get("error"):
            continue
        results.append((w_entry, rp, rpp))
    return results


# ── Reproduction table ────────────────────────────────────────────────────────

def print_cluster_table(run_name: str, best_program: Path, cluster: dict,
                        rp: dict, rpp: dict) -> None:
    """Print a comparison table for one cluster's representative workload."""
    sigs  = ", ".join(cluster["signatures"])
    w     = cluster["representative"]["w"]
    c     = cluster["representative"].get("c", {})
    w_str = ", ".join(f"{k}={v}" for k, v in w.items())
    c_str = json.dumps(c) if c else "{}"
    root  = (cluster.get("root_cause") or "")[:300]

    t_p  = rp.get("time",  0.0)
    t_pp = rpp.get("time", 0.0)
    m_p  = rp.get("mem_bytes",  0) / (1024.0 * 1024.0)
    m_pp = rpp.get("mem_bytes", 0) / (1024.0 * 1024.0)
    t_ratio = t_pp / (t_p + 1e-9)
    m_ratio_str = f"{m_pp / m_p:>8.2f}x" if m_p > 0 else "     N/A"

    p_out  = rp.get("output")  or {}
    pp_out = rpp.get("output") or {}

    evolved_label = "P' (evolved)"
    print(f"\n  Cluster — {cluster['trigger_func']}")
    print(f"    Types : {sigs}  |  Witnesses: {cluster['size']}")
    print(f"    Workload   : {w_str}")
    print(f"    Config P'  : {c_str}")
    print(f"\n    {'':20s} {'P (baseline)':>18s}  {evolved_label:>18s}  {'delta':>10s}")
    print(f"    {'Time':20s} {t_p:>15.3f} s  {t_pp:>15.3f} s  {t_ratio:>8.2f}x")
    print(f"    {'Memory':20s} {m_p:>13.1f} MB  {m_pp:>13.1f} MB  {m_ratio_str:>10s}")
    if p_out:
        out_p  = json.dumps(p_out)[:40]
        out_pp = json.dumps(pp_out)[:40]
        print(f"    {'Output':20s} {out_p:>18s}  {out_pp:>18s}  {'—':>10s}")
    if root:
        print(f"\n    Root cause: {root}")


def reproduce_run(run: dict, app_dir: Path) -> dict[str, tuple[dict, dict]]:
    """Print comparison tables for all clusters in one run.
    Returns {trigger_func: (rp, rpp)} for reuse in plotting."""
    width = 64
    print("\n" + "═" * width)
    print(f"  Run: {run['run_name']}")
    print(f"  P':  {run['best_program']}")
    print("═" * width)

    rw_code = (app_dir / "run_workload.py").read_text()
    initial = app_dir / "initial_program.py"
    best    = run["best_program"]
    n       = len(run["clusters"])
    rep_results: dict[str, tuple[dict, dict]] = {}

    for i, cluster in enumerate(run["clusters"], 1):
        print(f"\n  [{i}/{n}] re-running representative for {cluster['trigger_func']}...")
        rp, rpp = rerun_pair(initial, best, rw_code,
                             cluster["representative"]["w"], app_dir)
        if rp.get("error") or rpp.get("error"):
            print(f"  WARNING: re-run failed "
                  f"(P: {rp.get('error')}  P': {rpp.get('error')})")
            continue
        print_cluster_table(run["run_name"], best, cluster, rp, rpp)
        rep_results[cluster["trigger_func"]] = (rp, rpp)

    return rep_results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cluster(ax, rerun_data: list[tuple[dict, dict, dict]],
                 x_param: str | None, y_field: str, y_unit: str,
                 signatures: list[str]) -> None:
    """Draw P and P' series on ax for one cluster.

    Witnesses with the same x value are aggregated so that varying other
    parameters doesn't produce multiple y points at the same x position.
    scalab_time uses max (worst-case); all other metrics use median.
    """
    if not rerun_data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    xs   = [w["w"].get(x_param, i) if x_param else i
            for i, (w, _, _) in enumerate(rerun_data)]
    y_p  = [extract_y_value(rp,  y_field) for _, rp,  _   in rerun_data]
    y_pp = [extract_y_value(rpp, y_field) for _, _,   rpp in rerun_data]

    valid = [(x, yp, ypp) for x, yp, ypp in zip(xs, y_p, y_pp)
             if yp is not None and ypp is not None]
    if not valid:
        ax.text(0.5, 0.5, "no valid data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    use_max = "scalab_time" in signatures and y_field == "time"
    agg = np.max if use_max else np.median

    x_to_yp: dict = defaultdict(list)
    x_to_ypp: dict = defaultdict(list)
    for x, yp, ypp in valid:
        x_to_yp[x].append(yp)
        x_to_ypp[x].append(ypp)
    xs_s   = sorted(x_to_yp.keys())
    y_p_s  = [float(agg(x_to_yp[x]))  for x in xs_s]
    y_pp_s = [float(agg(x_to_ypp[x])) for x in xs_s]

    style = "-o" if len(xs_s) >= 3 else "o"
    ax.plot(xs_s, y_p_s,  style, color="steelblue", label="P (baseline)",
            markersize=5, linewidth=1.2)
    ax.plot(xs_s, y_pp_s, style, color="firebrick", label="P' (evolved)",
            markersize=5, linewidth=1.2)

    ax.set_xlabel(x_param or "witness index")
    ax.set_ylabel(y_unit)
    ax.legend(fontsize=7)


def plot_all(all_cluster_data: list[dict], out_path: Path,
             x_params: list[str], sig_filter: str | None) -> None:
    """Create figure with rows=(cluster × y-metric) × cols=x_params.

    Layout:
      - Each row is a (cluster, bug-type) pair — one row per plottable signature
        per cluster (correctness excluded; scalab_time/scalab_mem/optimality each
        get their own row so the user can compare P vs P' on every metric).
      - Each column is a workload parameter used as the x-axis.

    all_cluster_data items:
      {"cluster": dict, "run_name": str, "rerun_data": list[tuple],
       "rep_rp": dict|None, "rep_rpp": dict|None,
       "witnesses": list[dict]}
    """
    # Filter clusters by sig
    filtered = []
    for item in all_cluster_data:
        witnesses = item.get("witnesses", item["cluster"]["witnesses"])
        if sig_filter:
            witnesses = [w for w in witnesses if sig_filter in w.get("signatures", [])]
            if not witnesses:
                continue
        filtered.append({**item, "filtered_witnesses": witnesses})

    if not filtered:
        print("No clusters to plot after filtering.", file=sys.stderr)
        return

    if not x_params:
        print("No x-axis params available (no grammar.json found?).", file=sys.stderr)
        return

    # Build plot rows: one row per (cluster, plottable-sig) pair
    plot_rows = []
    for item in filtered:
        cluster = item["cluster"]
        rep_rp  = item.get("rep_rp")
        rep_rpp = item.get("rep_rpp")
        y_metrics = _y_metrics_for_cluster(cluster, rep_rp, rep_rpp)
        if not y_metrics:
            continue  # correctness-only cluster — nothing numeric to plot
        for sig, y_field, y_unit in y_metrics:
            rerun_data = item["rerun_data"]
            if sig_filter:
                rerun_data = [(w, rp, rpp) for w, rp, rpp in rerun_data
                              if sig_filter in w.get("signatures", [])]
            plot_rows.append({
                **item,
                "sig": sig, "y_field": y_field, "y_unit": y_unit,
                "rerun_data": rerun_data,
            })

    if not plot_rows:
        print("No plottable clusters (only correctness weaknesses?).", file=sys.stderr)
        return

    n_rows  = len(plot_rows)
    n_cols  = len(x_params)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for row_idx, row in enumerate(plot_rows):
        cluster  = row["cluster"]
        run_name = row["run_name"]
        sig      = row["sig"]
        y_field  = row["y_field"]
        y_unit   = row["y_unit"]
        rd       = row["rerun_data"]

        for col_idx, x_param in enumerate(x_params):
            ax = axes[row_idx][col_idx]
            ax.set_title(
                f"{run_name} / {cluster['trigger_func']}\n"
                f"x={x_param}  y={sig}  (N={cluster['size']})",
                fontsize=8)
            plot_cluster(ax, rd, x_param, y_field, y_unit, [sig])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Stored-metric plotting (no re-run) ────────────────────────────────────────

def _result_from_metrics(metrics: dict, suffix: str) -> dict:
    """Reshape stored Agent-2 metrics (time_p/time_pp/mem_p/...) into a
    run_one-style result dict so extract_y_value() can read it without re-running.
    suffix is 'p' (baseline P) or 'pp' (evolved P')."""
    return {
        "time":      metrics.get(f"time_{suffix}"),
        "mem_bytes": metrics.get(f"mem_{suffix}"),
        "output":    metrics.get(f"output_{suffix}") or {},
    }


def _stored_rerun_data(witnesses: list[dict]) -> list[tuple[dict, dict, dict]]:
    """Build (witness, rp, rpp) tuples from each witness's stored metrics, so the
    grid can be plotted with --max_witnesses 0 (no P/P' re-execution)."""
    out = []
    for w in witnesses:
        m = w.get("metrics")
        if not m:
            continue
        out.append((w, _result_from_metrics(m, "p"), _result_from_metrics(m, "pp")))
    return out


def _stored_pair(representative: dict) -> tuple[dict | None, dict | None]:
    """Representative (rp, rpp) from stored metrics — lets the optimality y-field
    chooser work in no-re-run mode."""
    m = representative.get("metrics")
    if not m:
        return None, None
    return _result_from_metrics(m, "p"), _result_from_metrics(m, "pp")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce bugs and plot P vs P' raw metrics.")
    parser.add_argument("results_dirs", nargs="+", type=Path)
    parser.add_argument("--x_param", default=None,
                        help="If given, plot only this param as x-axis (instead of all grammar params)")
    parser.add_argument("--sig", default=None,
                        choices=["scalab_time", "scalab_mem", "optimality", "correctness"],
                        help="Filter to one weakness type")
    parser.add_argument("--max_witnesses", type=int, default=50,
                        help="Max witnesses re-run per cluster "
                             "(0 = no re-run, plot stored metrics; default: 50)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: bug_plot.png in first results_dir)")
    parser.add_argument("--no_reproduce", action="store_true",
                        help="Skip Stage 1 comparison table")
    args = parser.parse_args()

    runs = [r for rd in args.results_dirs if (r := load_run(rd)) is not None]
    if not runs:
        sys.exit("No valid results directories found.")

    apps = {r["app"] for r in runs}
    if len(apps) > 1:
        sys.exit(f"Mixed apps: {apps}. All results_dirs must be for the same app.")

    app_dir = _ADRS_ROOT / apps.pop()
    if not app_dir.exists():
        print(f"WARNING: app dir {app_dir} not found; skipping re-runs.", file=sys.stderr)
        app_dir = None

    out_path = args.out or (args.results_dirs[0] / "bug_plot.png")
    rw_code  = (app_dir / "run_workload.py").read_text() if app_dir else ""
    initial  = (app_dir / "initial_program.py") if app_dir else None

    # Collect x-axis param list from grammar.json (union across runs, preserving order)
    if args.x_param:
        x_params = [args.x_param]
    else:
        seen: set[str] = set()
        x_params: list[str] = []
        for run in runs:
            for p in run.get("x_params", []):
                if p not in seen:
                    x_params.append(p)
                    seen.add(p)
        if not x_params:
            # Fallback: collect numeric params from first cluster's witnesses
            print("WARNING: no grammar.json found; inferring x-params from witness keys",
                  file=sys.stderr)
            for run in runs:
                for cluster in run["clusters"]:
                    if cluster["witnesses"]:
                        for k, v in cluster["witnesses"][0]["w"].items():
                            if isinstance(v, (int, float)) and k not in seen:
                                x_params.append(k)
                                seen.add(k)
                        break
                break

    # Collect per-cluster data (re-run representative + sampled witnesses)
    all_cluster_data = []
    for run in runs:
        if not args.no_reproduce and app_dir:
            rep_results = reproduce_run(run, app_dir)
        else:
            rep_results = {}

        for cluster in run["clusters"]:
            witnesses = cluster["witnesses"]
            if args.sig:
                witnesses = [w for w in witnesses if args.sig in w.get("signatures", [])]

            sampled  = sample_witnesses(witnesses, args.max_witnesses, None)

            rerun_data: list[tuple] = []
            rep_rp, rep_rpp = rep_results.get(cluster["trigger_func"], (None, None))

            if app_dir and initial and args.max_witnesses > 0:
                rerun_data = rerun_witnesses(sampled, initial, run["best_program"],
                                             rw_code, app_dir)
                if rep_rp is None:  # not yet run (--no_reproduce path)
                    rep_rp, rep_rpp = rerun_pair(initial, run["best_program"],
                                                 rw_code,
                                                 cluster["representative"]["w"], app_dir)
                    if rep_rp.get("error") or rep_rpp.get("error"):
                        rep_rp, rep_rpp = None, None
            else:
                # --max_witnesses 0 (or no app dir): plot the metrics already
                # stored on each witness during the Agent 2 oracle pass — no re-run.
                rerun_data = _stored_rerun_data(witnesses)
                if rep_rp is None:
                    rep_rp, rep_rpp = _stored_pair(cluster.get("representative", {}))

            all_cluster_data.append({
                "cluster":    cluster,
                "run_name":   run["run_name"],
                "rerun_data": rerun_data,
                "rep_rp":     rep_rp,
                "rep_rpp":    rep_rpp,
                "witnesses":  witnesses,  # already filtered by --sig
            })

    plot_all(all_cluster_data, out_path, x_params, args.sig)


if __name__ == "__main__":
    main()
