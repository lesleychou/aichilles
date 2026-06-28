"""
Agent 3: group confirmed bugs by trigger function, generate root-cause hypotheses.

run_agent3(app_dir, best_program_path, results_dir, client) -> dict
  Reads matrix_V.json from results_dir.
  Writes clusters.json and report.md to results_dir.
"""
import ast
import json
import re
from pathlib import Path

import anthropic
import numpy as np

from utils import render_template, MODEL

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# A Python traceback frame: `  File ".../best_program.py", line 319, in __init__`
_TB_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
# Last line of a traceback: `IndexError: list index out of range`
_TB_EXC_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt)): ?(.*)$", re.M)


def _crash_lines_from_traceback(text: str, prog_name: str) -> list[int]:
    """Pull the program's own crash line(s) out of a captured traceback.

    Returns the line numbers of frames whose file basename == prog_name (the
    program file), shallowest→deepest, so [-1] is the actual fault site. Empty if
    the text holds no traceback into the program file (e.g. a non-crash witness)."""
    if not text:
        return []
    lines = []
    for fpath, lineno, _func in _TB_FRAME_RE.findall(text):
        if Path(fpath).name == prog_name:
            lines.append(int(lineno))
    return lines


def _exception_summary(text: str) -> str:
    """The exception type + message from a traceback's last frame, e.g.
    'IndexError: list index out of range'. '' if none found."""
    if not text:
        return ""
    matches = _TB_EXC_RE.findall(text)
    if not matches:
        return ""
    etype, emsg = matches[-1]
    return f"{etype}: {emsg}".strip()


def _build_line_to_func(source: str) -> dict[int, str]:
    """Map each line number to its enclosing function name. Module-level → '__module__'."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    # collect (start_line, end_line, name) for every function def
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append((node.lineno, node.end_lineno, node.name))
    # sort by span length descending so narrower (inner) functions overwrite wider (outer) functions
    funcs.sort(key=lambda f: f[1] - f[0], reverse=True)
    mapping: dict[int, str] = {}
    for start, end, name in funcs:
        for ln in range(start, end + 1):
            mapping[ln] = name
    return mapping


def _trigger_func_from_v(v: list[float], line_to_func: dict[int, str]) -> str:
    """
    Identify the trigger function from a coverage vector v.
    Accumulates coverage weight per function; returns the function with the highest total.
    """
    arr = np.array(v)
    weights: dict[str, float] = {}
    for idx, val in enumerate(arr):
        if val <= 0:
            continue
        ln = idx + 1  # 1-based line number
        fn = line_to_func.get(ln, "__module__")
        weights[fn] = weights.get(fn, 0.0) + float(val)
    return max(weights, key=weights.__getitem__) if weights else "__module__"


def _top_lines_from_v(v: list[float], k: int = 10) -> list[int]:
    """Return top-k 1-based line numbers by coverage weight."""
    arr = np.array(v)
    indices = np.argsort(arr)[::-1][:k]
    return [int(i) + 1 for i in indices if arr[i] > 0]


_NO_CRASH_FUNC = "(no crash — quality/optimality regression)"


def group_by_trigger_func(entries: list[dict], pp_source: str, pp_name: str = "") -> list[dict]:
    """
    Group BUG entries by their trigger function. Preferred signal is the captured
    crash TRACEBACK (its deepest frame in the program file is the true fault site);
    this is used for subprocess-trained apps (e.g. nanochat) whose sys.settrace
    coverage is uninformative. Falls back to coverage `v` for legacy apps.
    Returns list of group dicts, each with:
      trigger_func, size, representative (highest delta), entries, anomalous_lines, signatures
    Sorted by size descending.
    """
    bug_entries = [e for e in entries if e.get("label") == "BUG"]
    if not bug_entries:
        return []

    line_to_func = _build_line_to_func(pp_source)
    groups: dict[str, list[dict]] = {}

    for e in bug_entries:
        tb = e.get("error_pp", "") or e.get("traceback_pp", "")
        crash_lines = _crash_lines_from_traceback(tb, pp_name) if pp_name else []
        if crash_lines:
            # Real fault site from the traceback's deepest program frame.
            func = line_to_func.get(crash_lines[-1], "__module__")
            e["_anomalous_lines"] = crash_lines[::-1]  # deepest (fault) first
        elif "error_pp" in e:
            # Traceback-style app, but this witness is a non-crash (optimality)
            # regression — there is no single buggy line.
            func = _NO_CRASH_FUNC
            e["_anomalous_lines"] = []
        else:
            # Legacy coverage-based app.
            v = e.get("v", [])
            func = _trigger_func_from_v(v, line_to_func)
            e["_anomalous_lines"] = _top_lines_from_v(v)
        groups.setdefault(func, []).append(e)

    result = []
    for func, members in groups.items():
        representative = max(members, key=lambda e: e.get("delta", 0.0))
        anomalous_lines = representative.get("_anomalous_lines",
                                             _top_lines_from_v(representative.get("v", [])))
        sigs: list[str] = []
        for m in members:
            for s in m.get("signatures", []):
                if s not in sigs:
                    sigs.append(s)
        result.append({
            "trigger_func": func,
            "size": len(members),
            "representative": representative,
            "entries": members,
            "anomalous_lines": anomalous_lines,
            "signatures": sigs,
        })
    result.sort(key=lambda g: g["size"], reverse=True)
    return result


def _annotate_source(source: str, anomalous_lines: list[int]) -> str:
    """Prefix anomalous lines with >>> in source."""
    lines = source.splitlines()
    anomalous_set = set(anomalous_lines)
    annotated = []
    for i, line in enumerate(lines, start=1):
        prefix = ">>> " if i in anomalous_set else "    "
        annotated.append(f"{prefix}{i:4d}  {line}")
    return "\n".join(annotated)


def _get_source_lines(source: str, line_numbers: list[int]) -> list[str]:
    """Return formatted (lineno, text) pairs for the given 1-based line numbers."""
    all_lines = source.splitlines()
    result = []
    for ln in line_numbers:
        idx = ln - 1
        if 0 <= idx < len(all_lines):
            result.append(f"  {ln:4d}: {all_lines[idx]}")
    return result


def _llm_root_cause(group: dict, pp_source: str, client: anthropic.Anthropic) -> str:
    rep = group["representative"]
    anomalous = group["anomalous_lines"]
    source_lines = pp_source.splitlines()

    # Lead the evidence with the actual exception + fault site from the traceback
    # (when the witness crashed); fall back to listing the anomalous source lines.
    tb = rep.get("error_pp", "") or rep.get("traceback_pp", "")
    exc = _exception_summary(tb)
    summary_parts = []
    if exc:
        site = f" at line {anomalous[0]}" if anomalous else ""
        summary_parts.append(f"P' raised: {exc}{site}")
    summary_parts += [
        f"line {ln}: {source_lines[ln-1].strip()}"
        for ln in anomalous[:10]
        if 0 < ln <= len(source_lines)
    ]
    if not summary_parts:
        summary_parts.append("(no crash — P' produced a worse-quality result than P)")

    prompt = render_template(_TEMPLATES_DIR / "agent3_rootcause.txt", {
        "pp_source_annotated": _annotate_source(pp_source, anomalous),
        "c_json": json.dumps(rep.get("c", {}), indent=2),
        "w_json": json.dumps(rep.get("w", {}), indent=2),
        "signature": ", ".join(group["signatures"]),
        "delta": f"{rep.get('delta', 0.0):.4f}",
        "anomalous_lines_summary": "\n".join(summary_parts),
    })

    response = client.messages.create(
        model=MODEL, max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


_HEAVY_KEYS = frozenset({"weights_flat", "weight", "load_tensor", "tokens"})


def _compact_w(w: dict) -> str:
    compact = {k: v for k, v in w.items()
               if k not in _HEAVY_KEYS and not (isinstance(v, list) and len(v) > 8)}
    return ", ".join(f"{k}={v}" for k, v in compact.items())


def _write_report(groups: list[dict], pp_source: str, results_dir: Path) -> None:
    lines = ["# Weakness Discovery Report\n"]

    lines += [
        "| # | Type | Trigger function | Witnesses | Example workload | Root cause |",
        "|---|-----------|-----------------|-----------|-----------------|------------|",
    ]
    for i, g in enumerate(groups, start=1):
        sig = ", ".join(g["signatures"])
        func = g["trigger_func"]
        top_lines = g["anomalous_lines"][:3]
        trigger_cell = f"`{func}` (lines {top_lines})"
        rep_w = _compact_w(g["representative"].get("w", {}))
        root = g.get("root_cause", "")
        short_root = root.split(".")[0].strip() if root else "(not analyzed)"
        lines.append(f"| {i} | {sig} | {trigger_cell} | {g['size']} | {rep_w} | {short_root} |")
    lines.append("")

    for i, g in enumerate(groups, start=1):
        rep = g["representative"]
        rep_w_full = {k: v for k, v in rep.get("w", {}).items() if k not in _HEAVY_KEYS}
        func = g["trigger_func"]
        top_lines = g["anomalous_lines"][:10]
        source_snippets = _get_source_lines(pp_source, top_lines)

        exc = _exception_summary(rep.get("error_pp", "") or rep.get("traceback_pp", ""))
        lines += [
            f"## Root Cause {i} — {', '.join(g['signatures'])} ({g['size']} witnesses)\n",
            f"**Trigger function:** `{func}`\n",
        ]
        if exc:
            lines.append(f"**Exception:** `{exc}` (from P' traceback)\n")
        lines.append("**Anomalous lines:**" if source_snippets
                     else "**Anomalous lines:** (none — non-crash quality regression)")
        lines.extend(source_snippets)
        lines += [
            "",
            f"**Root cause:** {g.get('root_cause', '(not analyzed)')}\n",
            f"**Highest-delta representative:**",
            f"- Config: `{json.dumps(rep.get('c', {}))}`",
            f"- Workload: `{json.dumps(rep_w_full)}`",
            f"- Delta: {rep.get('delta', 0.0):.4f}",
            "\n**Witnesses:**",
        ]
        for j, e in enumerate(g["entries"]):
            e_w = {k: v for k, v in e.get("w", {}).items() if k not in _HEAVY_KEYS}
            lines.append(
                f"  {j+1}. w={json.dumps(e_w)}, "
                f"delta={e.get('delta', 0):.3f}, types={e.get('signatures', [])}"
            )
        lines.append("")
    (results_dir / "report.md").write_text("\n".join(lines))


def run_agent3(
    app_dir: Path,
    best_program_path: Path,
    results_dir: Path,
    client: anthropic.Anthropic,
) -> dict:
    matrix_path = results_dir / "matrix_V.json"
    if not matrix_path.exists():
        print(f"[agent3] ERROR: {matrix_path} not found. Run agent2 first.")
        return {}

    entries = json.loads(matrix_path.read_text())
    bug_entries = [e for e in entries if e.get("label") == "BUG"]
    print(f"[agent3] {len(bug_entries)} weaknesses found in matrix V")

    if not bug_entries:
        (results_dir / "clusters.json").write_text("[]")
        (results_dir / "report.md").write_text("# Weakness Discovery Report\n\nNo weaknesses found.\n")
        return {"groups": 0}

    pp_source = best_program_path.read_text()
    groups = group_by_trigger_func(entries, pp_source, best_program_path.name)
    print(f"[agent3] {len(groups)} trigger-function groups found")

    for g in groups:
        print(f"  group '{g['trigger_func']}': {g['size']} weaknesses, "
              f"types={g['signatures']}")

    for g in groups:
        g["root_cause"] = _llm_root_cause(g, pp_source, client)
        print(f"  [{g['trigger_func']}] {g['root_cause'][:80]}...")

    def _strip(e: dict) -> dict:
        return {k: v for k, v in e.items() if k not in ("v", "v_delta", "v_norm")}

    serializable = [
        {
            "trigger_func": g["trigger_func"],
            "signatures": g["signatures"],
            "size": g["size"],
            "representative": _strip(g["representative"]),
            "witnesses": [_strip(e) for e in g["entries"]],
            "anomalous_lines": g["anomalous_lines"],
            "root_cause": g["root_cause"],
        }
        for g in groups
    ]

    (results_dir / "clusters.json").write_text(json.dumps(serializable, indent=2, default=str))
    _write_report(groups, pp_source, results_dir)
    print(f"[agent3] wrote clusters.json and report.md")
    return {"groups": len(groups)}
