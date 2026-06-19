"""
Differential oracle for risk discovery.

check(result_P, result_Pp, signature, **thresholds) -> OracleResult
  result_P / result_Pp: dicts with keys {output, time, rss, error, traceback}
    output: dict returned by run_workload (or None if program failed)
    time:   wall-clock seconds (float)
    rss:    peak resident set size in bytes (int); 0 if unavailable
    error:  error string or None

Optimality oracle checks all quality numeric fields in the output dict (excluding
time-related fields like speed_score, elapsed, runtime — those belong to scalab_time).
It fires if initial beats best by > thresh on any quality field
(relative delta: |P - P'| / (max(|P|,|P'|) + ε)). This avoids combined_score masking
component regressions (e.g. speed gains hiding balance degradation).
"""
from dataclasses import dataclass
from enum import Enum


class Signature(str, Enum):
    CORRECTNESS = "correctness"
    SCALAB_TIME = "scalab_time"
    SCALAB_MEM  = "scalab_mem"
    OPTIMALITY  = "optimality"

@dataclass
class OracleResult:
    is_witness: bool
    delta: float
    reason: str
    triggering_field: str | None = None

TIME_RATIO_THRESH  = 1.5
MEM_RATIO_THRESH   = 2.0
SCORE_DELTA_THRESH = 0.05
DEFAULT_TIMEOUT    = 30

def check(
    result_P:  dict,
    result_Pp: dict,
    signature: Signature,
    *,
    time_ratio_thresh:  float = TIME_RATIO_THRESH,
    mem_ratio_thresh:   float = MEM_RATIO_THRESH,
    score_delta_thresh: float = SCORE_DELTA_THRESH,
    timeout:            int   = DEFAULT_TIMEOUT,
) -> OracleResult:
    if signature == Signature.CORRECTNESS:
        return _check_correctness(result_P, result_Pp, timeout)
    if signature == Signature.SCALAB_TIME:
        return _check_scalab_time(result_P, result_Pp, time_ratio_thresh)
    if signature == Signature.SCALAB_MEM:
        return _check_scalab_mem(result_P, result_Pp, mem_ratio_thresh)
    if signature == Signature.OPTIMALITY:
        return _check_optimality(result_P, result_Pp, score_delta_thresh)
    raise ValueError(f"Unknown signature: {signature}")


def _check_correctness(p: dict, pp: dict, timeout: int) -> OracleResult:
    p_ok  = p.get("error") is None
    pp_ok = pp.get("error") is None
    if not p_ok or pp_ok:
        return OracleResult(is_witness=False, delta=0.0,
                            reason="both succeeded or P also failed")
    if pp["error"] == "timeout":
        delta = 1.0
        reason = f"P' timed out after {timeout}s; P succeeded"
    else:
        delta = 1.0
        reason = f"P' crashed: {pp['error']!r}"
    return OracleResult(is_witness=True, delta=delta, reason=reason)


def _check_scalab_time(p: dict, pp: dict, thresh: float) -> OracleResult:
    if p.get("error") or pp.get("error"):
        return OracleResult(is_witness=False, delta=0.0,
                            reason="skipped — one program errored")
    t_p  = p["time"]
    t_pp = pp["time"]
    delta = t_pp / (t_p + 1e-6)
    if delta > thresh:
        return OracleResult(
            is_witness=True, delta=delta,
            reason=f"P' is {delta:.1f}x slower than P ({t_pp:.2f}s vs {t_p:.2f}s)",
        )
    return OracleResult(is_witness=False, delta=delta,
                        reason=f"ratio {delta:.2f} below threshold {thresh}")


def _check_scalab_mem(p: dict, pp: dict, thresh: float) -> OracleResult:
    # Use tracemalloc peak (mem_bytes) — measures only allocations inside
    # run_workload(), immune to fork-inherited baseline memory.
    mem_p  = p.get("mem_bytes",  0)
    mem_pp = pp.get("mem_bytes", 0)
    if mem_p == 0 or mem_pp == 0 or p.get("error") or pp.get("error"):
        return OracleResult(is_witness=False, delta=0.0,
                            reason="mem unavailable or program errored")
    delta = mem_pp / (mem_p + 1)
    if delta > thresh:
        return OracleResult(
            is_witness=True, delta=delta,
            reason=(f"P' allocates {delta:.1f}x more peak memory "
                    f"({mem_pp//1024}KB vs {mem_p//1024}KB)"),
        )
    return OracleResult(is_witness=False, delta=delta,
                        reason=f"memory ratio {delta:.2f} below threshold {thresh}")


# Field name substrings that indicate raw algorithm execution-time measurements.
# These overlap with scalab_time and are excluded from optimality.
# NOTE: only exclude unambiguous raw-timing names. "speed_score", "throughput_score",
# "latency_score" etc. are OUTPUT QUALITY metrics and must stay in optimality.
# "times_" catches fields like times_algorithm, times_inference (raw wall-clock durations).
_TIME_FIELD_PATTERNS = ("elapsed", "runtime", "wall_time", "exec_time", "cpu_time", "times_")


def _is_time_field(name: str) -> bool:
    low = name.lower()
    return any(pat in low for pat in _TIME_FIELD_PATTERNS)


def _check_optimality(p: dict, pp: dict, thresh: float) -> OracleResult:
    # Crashes are correctness bugs, not optimality regressions — skip if either side crashed.
    if p.get("error") or pp.get("error"):
        return OracleResult(is_witness=False, delta=0.0,
                            reason="skipped: crash detected (correctness signature handles crashes)")

    out_p  = p.get("output")  if isinstance(p.get("output"),  dict) else {}
    out_pp = pp.get("output") if isinstance(pp.get("output"), dict) else {}
    crashed = False  # both sides confirmed non-crashed above

    # Collect quality (non-time) numeric fields from P's output.
    # Time-related fields (speed_score, elapsed, etc.) are handled by scalab_time.
    numeric_fields = [
        k for k, v in out_p.items()
        if isinstance(v, (int, float)) and not _is_time_field(k)
    ]
    if not numeric_fields:
        return OracleResult(is_witness=False, delta=0.0,
                            reason="no numeric fields in P output")

    best_field: str | None = None
    best_rel_delta = 0.0

    for field in numeric_fields:
        v_p  = out_p[field]
        v_pp = out_pp.get(field)
        if v_pp is None or not isinstance(v_pp, (int, float)):
            continue
        # Only fire when P is strictly better than P' (v_p > v_pp).
        # We assume quality fields are higher-is-better. Lower-is-better raw timing
        # fields (elapsed, times_, etc.) are already excluded above and handled by
        # the scalab_time oracle instead.
        if v_p <= v_pp:
            continue
        rel_delta = (v_p - v_pp) / (max(abs(v_p), abs(v_pp)) + 1e-6)
        if rel_delta > thresh and rel_delta > best_rel_delta:
            best_rel_delta = rel_delta
            best_field = field

    if best_field is not None:
        v_p  = out_p[best_field]
        v_pp = out_pp.get(best_field, 0.0)
        return OracleResult(
            is_witness=True, delta=best_rel_delta,
            reason=(f"field '{best_field}': P={v_p:.4f}, P'={v_pp:.4f}"
                    f" (rel_delta={best_rel_delta:.4f})"),
            triggering_field=best_field,
        )

    # Not a witness — report max directional delta (P > P') seen across all fields
    max_rel = 0.0
    for field in numeric_fields:
        v_p  = out_p[field]
        v_pp = out_pp.get(field)
        if v_pp is None or not isinstance(v_pp, (int, float)):
            continue
        if v_p <= v_pp:
            continue
        rel = (v_p - v_pp) / (max(abs(v_p), abs(v_pp)) + 1e-6)
        max_rel = max(max_rel, rel)

    return OracleResult(is_witness=False, delta=max_rel,
                        reason=f"no field exceeded rel threshold {thresh}")
