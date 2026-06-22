# aichilles_risk_discovery/demo/export_demo.py
"""
Export real weakness-discovery results into the interactive demo HTML.

Reads every results/<app>/<run>/ directory (config.json + clusters.json + grammar.json),
builds a compact JSON, and injects it into demo/explorer_template.html to produce a
self-contained, double-clickable explorer.html.

Usage:
  python aichilles_risk_discovery/demo/export_demo.py            # uses local results/

  # Publish the GitHub Pages demo from the full paper results:
  python aichilles_risk_discovery/demo/export_demo.py \
      --results /path/to/skydiscover/benchmarks/ADRS/risk_discovery_v2/new_results \
      --out docs/explorer/index.html
"""
import argparse
import json
import re
import statistics
from pathlib import Path

_HERE = Path(__file__).parent
_DEFAULT_RESULTS = _HERE.parent / "results"
_TEMPLATE = _HERE / "explorer_template.html"

# best/<model>/<method>/best_program.py  →  display names
_LLM_NAMES  = {"claude": "Claude-Opus-4.6", "gpt": "GPT-5"}
_ALGO_NAMES = {"adaevolve": "AdaEvolve", "engram": "Engram", "openevolve": "OpenEvolve"}

# Priority order for picking a cluster's "primary" plottable type.
_PLOTTABLE = ["scalab_time", "scalab_mem", "optimality"]


def _result(metrics: dict, suffix: str) -> dict:
    """Stored Agent-2 metrics → {time, mem_mb, output}."""
    mem = metrics.get(f"mem_{suffix}")
    return {
        "time":   metrics.get(f"time_{suffix}"),
        "mem_mb": (mem / 1_048_576.0) if isinstance(mem, (int, float)) else None,
        "output": metrics.get(f"output_{suffix}") or {},
    }


def _y_for_type(t: str, rep_w: dict, witnesses: list) -> tuple[str, str]:
    """Return (metric_field, y_unit) for a plottable type. For optimality, pick the
    representative output field with the largest relative P-vs-P' gap."""
    if t == "scalab_time":
        return "time", "seconds"
    if t == "scalab_mem":
        return "mem_mb", "MB"
    # optimality: choose the most divergent numeric output field
    rep = witnesses[0]["metrics"] if witnesses else {}
    p_out, pp_out = rep.get("output_p") or {}, rep.get("output_pp") or {}
    best_field, best_gap = None, -1.0
    for f, pv in p_out.items():
        ppv = pp_out.get(f)
        if isinstance(pv, (int, float)) and isinstance(ppv, (int, float)):
            gap = abs(pv - ppv) / (max(abs(pv), abs(ppv)) + 1e-9)
            if gap > best_gap:
                best_gap, best_field = gap, f
    return (best_field or "time"), (best_field or "score")


def _extract_y(result: dict, field: str):
    if field in ("time", "mem_mb"):
        return result.get(field)
    v = (result.get("output") or {}).get(field)
    return float(v) if isinstance(v, (int, float)) else None


def _pick_x_param(params: list, witnesses: list) -> str | None:
    """Numeric workload param (excluding 'seed') with the most distinct witness values."""
    best, best_n = None, 0
    for p in params:
        if p == "seed":
            continue
        vals = {w["w"].get(p) for w in witnesses if isinstance(w["w"].get(p), (int, float))}
        if len(vals) > best_n:
            best, best_n = p, len(vals)
    return best


def _build_curve(cluster: dict, primary_type: str, params: list) -> dict | None:
    witnesses = cluster["witnesses"]
    x_param = _pick_x_param(params, witnesses)
    if not x_param:
        return None
    y_field, y_unit = _y_for_type(primary_type, cluster["representative"]["w"], witnesses)
    agg = max if primary_type == "scalab_time" else statistics.median

    by_x_p: dict = {}
    by_x_pp: dict = {}
    for w in witnesses:
        x = w["w"].get(x_param)
        if not isinstance(x, (int, float)):
            continue
        yp  = _extract_y(_result(w["metrics"], "p"),  y_field)
        ypp = _extract_y(_result(w["metrics"], "pp"), y_field)
        if yp is None or ypp is None:
            continue
        by_x_p.setdefault(x, []).append(yp)
        by_x_pp.setdefault(x, []).append(ypp)

    xs = sorted(by_x_p)
    if not xs:
        return None
    points = [{"x": x,
               "initial": round(float(agg(by_x_p[x])), 4),
               "evolved": round(float(agg(by_x_pp[x])), 4)} for x in xs]
    return {"xParam": x_param, "yUnit": y_unit, "yField": y_field, "points": points}


_LINE_MENTION = re.compile(r'lines?\s+(\d[\d\s,\-–]*(?:and\s+\d+)?[\d\s,\-–]*)', re.I)


def _lines_from_text(text: str, n: int) -> list:
    """Extract P' line numbers/ranges cited in the LLM root-cause prose, e.g.
    'lines 70-116', 'line 78', 'lines 78 and 91'. Used when coverage gave no
    anomalous_lines (common for __module__/top-level clusters)."""
    found: set = set()
    for m in _LINE_MENTION.finditer(text or ""):
        chunk = m.group(1)
        for a, b in re.findall(r'(\d+)\s*[-–]\s*(\d+)', chunk):
            a, b = int(a), int(b)
            if a <= b and b - a <= 200:
                found.update(range(a, b + 1))
        for num in re.findall(r'\d+', re.sub(r'\d+\s*[-–]\s*\d+', ' ', chunk)):
            found.add(int(num))
    return sorted(x for x in found if 1 <= x <= n)


def _highlight_lines(src_lines: list, trigger_func: str, anomalous: list,
                     root_cause: str = "") -> list:
    """1-based line numbers in P' (best_program.py) to highlight as the weakness
    region: the delta-coverage anomalous lines, else line numbers cited in the
    root-cause text, else the trigger function's span."""
    n = len(src_lines)
    hl = sorted({a for a in anomalous if isinstance(a, int) and 1 <= a <= n})
    if hl:
        return hl
    from_text = _lines_from_text(root_cause, n)
    if from_text:
        return from_text
    if trigger_func and trigger_func != "__module__":
        for i, ln in enumerate(src_lines):
            if ln.lstrip().startswith(f"def {trigger_func}("):
                indent = len(ln) - len(ln.lstrip())
                end = n
                for j in range(i + 1, n):
                    s = src_lines[j]
                    if s.strip() and (len(s) - len(s.lstrip())) <= indent and \
                       s.lstrip().startswith(("def ", "class ")):
                        end = j
                        break
                return list(range(i + 1, end + 1))
    return []


def _cell_from_run(run_dir: Path, initial_by_app: dict) -> dict | None:
    cfg_p, cl_p, gr_p = (run_dir / "config.json", run_dir / "clusters.json", run_dir / "grammar.json")
    if not (cfg_p.exists() and cl_p.exists()):
        return None
    cfg = json.loads(cfg_p.read_text())
    clusters = json.loads(cl_p.read_text())
    params = [p["name"] for p in (json.loads(gr_p.read_text()).get("grammar_workload", []) if gr_p.exists() else [])]

    bp = Path(cfg["best_program"])
    parts = bp.parts
    bidx = parts.index("best") if "best" in parts else None
    model = parts[bidx + 1] if bidx is not None else "claude"
    algo  = parts[bidx + 2] if bidx is not None else "adaevolve"
    app   = cfg["app"]

    # Full evolved P' source (per cell), and full baseline P source (deduped per app).
    evolved = bp.read_text().splitlines() if bp.exists() else ["(best_program.py not found)"]
    if app not in initial_by_app:
        ip = (Path(*parts[:bidx]) / "initial_program.py") if bidx is not None else None
        initial_by_app[app] = ip.read_text().splitlines() if (ip and ip.exists()) \
            else ["(initial_program.py not found)"]

    weaknesses = []
    for i, c in enumerate(clusters):
        types = c.get("signatures", [])
        primary = next((t for t in _PLOTTABLE if t in types), None)
        evidence = "curve" if primary else "crash"
        curve = _build_curve(c, primary, params) if primary else None
        if primary and not curve:
            evidence = "crash"  # fall back if no plottable x
        weaknesses.append({
            "id": f"{app}-{i}",
            "name": f"{(primary or 'correctness')} · {c['trigger_func']}",
            "triggerFunc": c["trigger_func"],
            "types": types,
            "size": c["size"],
            "delta": round(float(c["representative"].get("delta", 0.0)), 3),
            "rootCause": c.get("root_cause", ""),
            "workload": {k: v for k, v in c["representative"]["w"].items() if k != "seed"} or c["representative"]["w"],
            "evidence": evidence,
            "curve": curve,
            "highlight": _highlight_lines(evolved, c["trigger_func"], c.get("anomalous_lines", []),
                                          c.get("root_cause", "")),
        })

    return {
        "app": app,
        "algo": _ALGO_NAMES.get(algo, algo),
        "llm": _LLM_NAMES.get(model, model),
        "run": run_dir.name,
        "params": [p for p in params if p != "seed"],
        "evolved": evolved,
        "weaknesses": weaknesses,
    }


def build_data(results_root: Path) -> dict:
    # Latest run per (app, algo, llm) cell.
    latest: dict[tuple, Path] = {}
    for cfg_p in results_root.glob("*/*/config.json"):
        run_dir = cfg_p.parent
        try:
            cfg = json.loads(cfg_p.read_text())
            bp_parts = Path(cfg["best_program"]).parts
            model = bp_parts[bp_parts.index("best") + 1]
            algo  = bp_parts[bp_parts.index("best") + 2]
        except Exception:
            continue
        key = (cfg["app"], algo, model)
        if key not in latest or run_dir.name > latest[key].name:  # name has the timestamp
            latest[key] = run_dir

    initial_by_app: dict = {}
    cells = [c for rd in latest.values() if (c := _cell_from_run(rd, initial_by_app))]
    cells.sort(key=lambda c: (c["app"], c["algo"], c["llm"]))
    return {"cells": cells, "initialByApp": initial_by_app}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=_DEFAULT_RESULTS)
    ap.add_argument("--template", type=Path, default=_TEMPLATE)
    ap.add_argument("--out", type=Path, default=_HERE / "explorer.html")
    args = ap.parse_args()

    data = build_data(args.results)
    n_cells = len(data["cells"])
    n_weak = sum(len(c["weaknesses"]) for c in data["cells"])
    if n_cells == 0:
        print("WARNING: no result cells found under", args.results)

    template = args.template.read_text()
    marker = "/*__ADRS_DATA__*/null"
    if marker not in template:
        raise SystemExit(f"template missing data marker {marker!r}")
    html = template.replace(marker, json.dumps(data, separators=(",", ":")))
    args.out.write_text(html)
    print(f"[export_demo] {n_cells} cells, {n_weak} weakness clusters → {args.out}")


if __name__ == "__main__":
    main()
