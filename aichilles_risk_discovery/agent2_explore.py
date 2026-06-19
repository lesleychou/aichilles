# aichilles_risk_discovery/agent2_explore.py
"""
Agent 2: Type-specialized MAP-Elites bug exploration.

run_agent_type(sig, app_dir, best_program_path, results_dir, client,
               budget, patience, theta, crash_workloads) -> (dict, list)
  sig: Signature enum value — one of CORRECTNESS, SCALAB_TIME, SCALAB_MEM, OPTIMALITY
  Returns (summary_dict, crash_workloads_list).
  crash_workloads_list contains workloads where P' errored or timed out during this run.

  Reads grammar.json and generate_workload.py from results_dir (Agent 1 outputs).
  Writes matrix_V_{sig}.json (checkpointed each round) to results_dir.
  Appends to witnesses_{type}.json for every oracle type that fires.
"""
import json
import sys
from pathlib import Path

import anthropic
import numpy as np

from archive import (
    MapElitesArchive, MatrixV,
    coverage_to_vector, normalize_vector, compute_novelty,
)
from harness import run_one
from oracle import check, Signature
from utils import extract_code_block, render_template, MODEL, safe_exec

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_ALL_SIGNATURES = [
    Signature.CORRECTNESS, Signature.SCALAB_TIME,
    Signature.SCALAB_MEM, Signature.OPTIMALITY,
]
_DEFAULT_NOVELTY_THETA = 0.1
_TIMEOUT_SCALAB_DELTA = 30.0  # fixed delta for P' timeout treated as scalab_time witness

_SIG_TEMPLATE = {
    Signature.CORRECTNESS: "agent2_correctness.txt",
    Signature.SCALAB_TIME: "agent2_scalab_time.txt",
    Signature.SCALAB_MEM:  "agent2_scalab_mem.txt",
    Signature.OPTIMALITY:  "agent2_optimality.txt",
}


def _count_lines(program_path: Path) -> int:
    return len(program_path.read_text().splitlines())


def _grammar_param_info(grammar: dict) -> dict[str, str]:
    grammar_workload = grammar.get("grammar_workload", {})
    if isinstance(grammar_workload, list):
        return {spec["name"]: str(spec.get("range", "?"))
                for spec in grammar_workload if isinstance(spec, dict) and "name" in spec}
    return {k: f"[{spec.get('min')}, {spec.get('max')}]"
            for k, spec in grammar_workload.items()}


def _param_exploration_summary(tried_workloads: list[dict], grammar: dict) -> str:
    if not tried_workloads:
        return "(no exploration data yet)"
    param_info = _grammar_param_info(grammar)
    if not param_info:
        return "(no exploration data yet)"
    lines = []
    for key, g_range in param_info.items():
        vals = [w.get(key) for w in tried_workloads if isinstance(w.get(key), (int, float))]
        if not vals:
            continue
        lines.append(f"  {key}: grammar={g_range!r}, explored=[{min(vals)}, {max(vals)}]")
    return "\n".join(lines) if lines else "(no numeric parameters tracked)"


def _coverage_summary(v_norm: np.ndarray, top_k: int = 10) -> str:
    if v_norm.sum() == 0:
        return "(no coverage data)"
    indices = np.argsort(v_norm)[::-1][:top_k]
    parts = [f"line {i+1}: {v_norm[i]:.3f}" for i in indices if v_norm[i] > 0]
    return ", ".join(parts)


def _crash_warning(crash_workloads: list[dict], sig_value: str) -> str:
    """
    Format a warning about parameter values that cause P' to crash.
    Returns empty string for the correctness agent (crashes are desired there).
    """
    if sig_value == "correctness" or not crash_workloads:
        return ""
    crash_axes: dict[str, list] = {}
    for w in crash_workloads:
        for k, v in w.items():
            if isinstance(v, (int, float, str, bool)):
                crash_axes.setdefault(k, [])
                if v not in crash_axes[k]:
                    crash_axes[k].append(v)
    if not crash_axes:
        return ""
    lines = [
        "⚠ P' CRASHES on these parameter values (avoid — they give no differential signal here):",
    ]
    for k, vals in sorted(crash_axes.items()):
        sorted_vals = sorted(vals, key=lambda v: (isinstance(v, str), v))
        lines.append(f"  {k}: {sorted_vals[:8]}")
    return "\n".join(lines)


def _check_all_signatures(result_p: dict, result_pp: dict) -> tuple[list[str], float]:
    """Check all 4 oracles. Returns (fired_signature_values, max_delta)."""
    fired = []
    max_delta = 0.0
    for sig in _ALL_SIGNATURES:
        r = check(result_p, result_pp, sig)
        if r.is_witness:
            fired.append(sig.value)
        max_delta = max(max_delta, r.delta)
    return fired, max_delta


def _format_witness_examples(witnesses: list[dict], n: int = 5) -> str:
    if not witnesses:
        return "(none yet)"
    examples = witnesses[-n:]  # most recent n
    return "\n".join(json.dumps({"w": e.get("w", {}), "delta": e.get("delta", 0)})
                     for e in examples)


def _llm_mutate_for_sig(
    sig: Signature,
    seed_entry: dict,
    grammar: dict,
    tried_workloads: list[dict],
    crash_workloads: list[dict],
    witnesses: list[dict],
    client: anthropic.Anthropic,
) -> list[dict]:
    """Call LLM to mutate seed_entry into 5 candidate (c, w) pairs for this sig type."""
    v_norm = np.array(seed_entry.get("v_norm", []))
    template_file = _TEMPLATES_DIR / _SIG_TEMPLATE[sig]
    prompt = render_template(template_file, {
        "grammar_json": json.dumps(grammar, indent=2),
        "seed_cw_json": json.dumps({"c": seed_entry.get("c", {}),
                                     "w": seed_entry.get("w", {})}, indent=2),
        "coverage_summary": _coverage_summary(v_norm) if len(v_norm) > 0 else "(unavailable)",
        "param_exploration_summary": _param_exploration_summary(tried_workloads, grammar),
        "crash_warning": _crash_warning(crash_workloads, sig.value),
        "witness_examples": _format_witness_examples(witnesses),
    })

    response = client.messages.create(
        model=MODEL, max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    code = extract_code_block(response.content[0].text)
    if not code:
        return []
    result, err = safe_exec(code, "generate_workloads")
    if err or not isinstance(result, list):
        print(f"[agent2/{sig.value}] LLM mutation exec error: {err}", file=sys.stderr)
        return []
    return result


def _checkpoint_matrix(matrix_v: MatrixV, results_dir: Path, sig: Signature) -> None:
    path = results_dir / f"matrix_V_{sig.value}.json"
    path.write_text(json.dumps(matrix_v.all_entries(), indent=2, default=str))


def _checkpoint_witnesses(
    all_witnesses: dict[str, list],
    results_dir: Path,
) -> None:
    """Write all 4 witness files (merge new findings into existing file contents)."""
    for sig_value, witnesses in all_witnesses.items():
        path = results_dir / f"witnesses_{sig_value}.json"
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = []
        if not witnesses and existing:
            continue  # nothing new to add
        seen = {json.dumps(e, sort_keys=True, default=str) for e in existing}
        merged = existing + [e for e in witnesses
                             if json.dumps(e, sort_keys=True, default=str) not in seen]
        path.write_text(json.dumps(merged, indent=2, default=str))


def run_agent_type(
    sig: Signature,
    app_dir: Path,
    best_program_path: Path,
    results_dir: Path,
    client: anthropic.Anthropic,
    budget: int = 50,
    patience: int = 5,
    theta: float = _DEFAULT_NOVELTY_THETA,
    crash_workloads: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    """
    Run one type-specialized exploration agent for the given signature.

    Returns:
      (summary_dict, crash_workloads_list)
      crash_workloads_list contains workloads where P' errored or timed out during this run.
      Only meaningful for the CORRECTNESS agent; empty list for others.
    """
    if crash_workloads is None:
        crash_workloads = []

    grammar_path = results_dir / "grammar.json"
    generate_path = results_dir / "generate_workload.py"
    rw_path = app_dir / "run_workload.py"

    for path, label in [
        (grammar_path,  "grammar.json"),
        (generate_path, "generate_workload.py"),
        (rw_path,       "run_workload.py"),
    ]:
        if not path.exists():
            print(f"[agent2/{sig.value}] ERROR: {label} not found at {path}.", file=sys.stderr)
            return {}, []

    grammar = json.loads(grammar_path.read_text())
    generate_workload_code = generate_path.read_text()
    run_workload_code = rw_path.read_text() + "\n\n" + generate_workload_code

    initial_path = app_dir / "initial_program.py"
    n_lines = _count_lines(best_program_path)
    print(f"[agent2/{sig.value}] P' has {n_lines} lines. "
          f"Budget={budget}, patience={patience}, theta={theta}")

    archive = MapElitesArchive(bucket_k=5)
    matrix_v = MatrixV()

    # Per-run witness lists for all 4 types (this agent may find witnesses of any type)
    run_witnesses: dict[str, list] = {s.value: [] for s in _ALL_SIGNATURES}

    # New crash workloads discovered by this agent (returned to caller)
    new_crash_workloads: list[dict] = []

    # Warm-start: KB seeds filtered to this sig first, then fall back to all seeds
    from knowledge_base import load_knowledge_base
    kb_dir = results_dir.parent
    kb = load_knowledge_base(kb_dir)
    all_seeds = kb.get("bug_seeds", []) + kb.get("high_delta_seeds", [])
    sig_seeds = [s for s in all_seeds if sig.value in s.get("signatures", [])]
    warm_seeds = (sig_seeds or all_seeds)[:max(budget // 4, 5)]

    oracle_calls = 0
    if warm_seeds:
        print(f"[agent2/{sig.value}] warm-starting with {len(warm_seeds)} seeds")
        for seed in warm_seeds:
            if oracle_calls >= budget:
                break
            w = seed.get("w", {})
            if sig != Signature.CORRECTNESS and any(w == crash_wl for crash_wl in crash_workloads):
                continue
            result_pp = run_one(str(best_program_path), run_workload_code, w,
                                timeout=30, collect_coverage=True, app_dir=str(app_dir))
            if result_pp.get("error") and sig != Signature.CORRECTNESS:
                continue
            raw_counts = result_pp.get("coverage", {})
            v = coverage_to_vector(raw_counts, n_lines)
            v_norm = normalize_vector(v)
            # Run P' again without coverage for clean timing/memory oracle comparison.
            # The covered run inflates wall-clock time 2-5x via sys.settrace overhead,
            # causing systematic scalab_time false positives if used directly.
            result_pp_oracle = run_one(str(best_program_path), run_workload_code, w,
                                       timeout=30, collect_coverage=False, app_dir=str(app_dir))
            result_p = run_one(str(initial_path), run_workload_code, w,
                               timeout=30, collect_coverage=False, app_dir=str(app_dir))
            oracle_calls += 1
            fired_sigs, max_delta = _check_all_signatures(result_p, result_pp_oracle)
            label = "BUG" if fired_sigs else "NO_BUG"
            entry = {"c": seed.get("c", {}), "w": w, "v": v_norm.tolist(),
                     "label": label, "signatures": fired_sigs, "delta": max_delta,
                     "round": -1, "v_norm": v_norm.tolist(),
                     "metrics": {
                         "time_p":   result_p["time"],
                         "time_pp":  result_pp_oracle["time"],
                         "mem_p":    result_p.get("mem_bytes", 0),
                         "mem_pp":   result_pp_oracle.get("mem_bytes", 0),
                         "output_p":  result_p.get("output") or {},
                         "output_pp": result_pp_oracle.get("output") or {},
                     }}
            matrix_v.append(entry)
            if label == "BUG":
                archive.update(v_norm, entry)
                for s in fired_sigs:
                    run_witnesses[s].append(entry)
                print(f"  [warm] weakness found: types={fired_sigs}, delta={max_delta:.3f}")
            if result_pp.get("error"):
                new_crash_workloads.append(w)

    # Seed archive with a fresh workload if still empty after warm-start
    if archive.size() == 0:
        try:
            ns = {}
            exec(generate_workload_code, ns)  # noqa: S102
            seed_cw = ns["generate_workload"](grammar)
            _zero = np.zeros(n_lines)
            archive.update(_zero, {**seed_cw, "delta": 0.0, "label": "NO_BUG",
                                   "signatures": [], "v_norm": _zero.tolist()})
        except Exception as exc:
            print(f"[agent2/{sig.value}] failed to seed archive: {exc}", file=sys.stderr)

    consecutive_empty = 0
    round_idx = 0

    while oracle_calls < budget and consecutive_empty < patience:
        print(f"[agent2/{sig.value}] round {round_idx} | oracle_calls={oracle_calls} | "
              f"weaknesses={matrix_v.bug_count()} | archive_size={archive.size()}")

        seed = archive.sample(prefer_signature=sig.value)
        if seed is None:
            break

        tried_workloads = [e["w"] for e in matrix_v.all_entries()]
        witnesses_so_far = run_witnesses[sig.value]
        candidates = _llm_mutate_for_sig(sig, seed, grammar, tried_workloads,
                                         crash_workloads + new_crash_workloads,
                                         witnesses_so_far, client)
        if not candidates:
            round_idx += 1
            consecutive_empty += 1
            continue

        new_bugs_this_round = 0

        for cw in candidates:
            if oracle_calls >= budget:
                break
            c = cw.get("c", {})
            w = cw.get("w", {})

            # Skip crash-known workloads for non-correctness agents
            if sig != Signature.CORRECTNESS and any(
                w == crash_wl for crash_wl in crash_workloads + new_crash_workloads
            ):
                continue

            # Step a: collect coverage from P'
            result_pp_cov = run_one(str(best_program_path), run_workload_code, w,
                                    timeout=30, collect_coverage=True, app_dir=str(app_dir))

            # Handle P' timeout as scalab_time witness
            if result_pp_cov.get("error") == "timeout":
                result_p_check = run_one(str(initial_path), run_workload_code, w,
                                         timeout=30, collect_coverage=False, app_dir=str(app_dir))
                oracle_calls += 1
                if result_p_check.get("error") is None:
                    timeout_entry = {
                        "c": c, "w": w, "v": np.zeros(n_lines).tolist(),
                        "label": "BUG", "signatures": ["scalab_time"],
                        "delta": _TIMEOUT_SCALAB_DELTA, "round": round_idx,
                        "v_norm": np.zeros(n_lines).tolist(),
                        "metrics": {
                            "time_p":   result_p_check["time"],
                            "time_pp":  float(30),  # timed out
                            "mem_p":    result_p_check.get("mem_bytes", 0),
                            "mem_pp":   0,
                            "output_p":  result_p_check.get("output") or {},
                            "output_pp": {},
                        },
                    }
                    matrix_v.append(timeout_entry)
                    run_witnesses["scalab_time"].append(timeout_entry)
                    new_bugs_this_round += 1
                    print(f"  [+] timeout→scalab_time witness, delta={_TIMEOUT_SCALAB_DELTA}")
                new_crash_workloads.append(w)
                continue

            # For non-correctness agents: skip if P' errored
            if result_pp_cov.get("error") and sig != Signature.CORRECTNESS:
                new_crash_workloads.append(w)
                continue

            raw_counts = result_pp_cov.get("coverage", {})
            v = coverage_to_vector(raw_counts, n_lines)
            v_norm = normalize_vector(v)

            # Coverage novelty filter (coverage-only — no param distance)
            if compute_novelty(v_norm, archive.all_vectors()) < theta:
                continue

            # Run P' without coverage for clean timing/memory oracle comparison.
            # The covered run inflates wall-clock time 2-5x via sys.settrace overhead,
            # causing systematic scalab_time false positives if used directly.
            result_pp_oracle = run_one(str(best_program_path), run_workload_code, w,
                                       timeout=30, collect_coverage=False, app_dir=str(app_dir))
            result_p = run_one(str(initial_path), run_workload_code, w,
                               timeout=30, collect_coverage=False, app_dir=str(app_dir))
            oracle_calls += 1

            fired_sigs, max_delta = _check_all_signatures(result_p, result_pp_oracle)
            label = "BUG" if fired_sigs else "NO_BUG"

            entry = {
                "c": c, "w": w,
                "v": v_norm.tolist(),
                "label": label,
                "signatures": fired_sigs,
                "delta": max_delta,
                "round": round_idx,
                "metrics": {
                    "time_p":   result_p["time"],
                    "time_pp":  result_pp_oracle["time"],
                    "mem_p":    result_p.get("mem_bytes", 0),
                    "mem_pp":   result_pp_oracle.get("mem_bytes", 0),
                    "output_p":  result_p.get("output") or {},
                    "output_pp": result_pp_oracle.get("output") or {},
                },
                "v_norm": v_norm.tolist(),
            }
            matrix_v.append(entry)
            archive.update(v_norm, entry)

            if label == "BUG":
                new_bugs_this_round += 1
                for s in fired_sigs:
                    run_witnesses[s].append(entry)
                print(f"  [+] weakness found: types={fired_sigs}, delta={max_delta:.3f}")

            if result_pp_cov.get("error"):
                new_crash_workloads.append(w)

        _checkpoint_matrix(matrix_v, results_dir, sig)
        _checkpoint_witnesses(run_witnesses, results_dir)
        consecutive_empty = 0 if new_bugs_this_round > 0 else consecutive_empty + 1
        round_idx += 1

    # Final checkpoint — ensures files are written even if the while loop never ran
    # (e.g. budget exhausted entirely during warm-start).
    _checkpoint_matrix(matrix_v, results_dir, sig)
    _checkpoint_witnesses(run_witnesses, results_dir)

    print(f"[agent2/{sig.value}] done. total_weaknesses={matrix_v.bug_count()}, "
          f"oracle_calls={oracle_calls}")
    return (
        {"total_weaknesses": matrix_v.bug_count(), "oracle_calls": oracle_calls, "rounds": round_idx},
        new_crash_workloads,
    )
