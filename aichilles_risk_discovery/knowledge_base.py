# aichilles_risk_discovery/knowledge_base.py
"""
Per-app knowledge base: read/write transferable knowledge across runs.

Files: results/{app}/knowledge_base.json
"""
import copy
import json
from pathlib import Path

def _empty_kb():
    return {
        "bug_seeds": [],
        "high_delta_seeds": [],
        "signature_priors": {},
        "discriminating_regions": [],
    }


def load_knowledge_base(app_kb_dir: Path) -> dict:
    """Load knowledge_base.json from app_kb_dir. Returns empty KB if not found."""
    kb_path = app_kb_dir / "knowledge_base.json"
    if not kb_path.exists():
        return _empty_kb()
    try:
        return json.loads(kb_path.read_text())
    except Exception:
        return _empty_kb()


def save_knowledge_base(results_dir: Path, app_name: str, kb: dict) -> None:
    """Write knowledge_base.json to results_dir/{app_name}/."""
    app_dir = results_dir / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "knowledge_base.json").write_text(json.dumps(kb, indent=2, default=str))


def extract_transferable(entries: list[dict], existing_kb: dict | None) -> dict:
    """
    Extract transferable knowledge from matrix V entries.
    Returns a new KB dict (does not modify existing_kb in place).
    """
    kb = _empty_kb()

    bug_entries = [e for e in entries if e.get("label") == "BUG"]
    no_bug_entries = [e for e in entries if e.get("label") == "NO_BUG"]

    # Bug seeds: all confirmed bugs
    for e in bug_entries:
        sigs = e.get("signatures", [])
        kb["bug_seeds"].append({
            "c": e.get("c", {}),
            "w": e.get("w", {}),
            "delta": e.get("delta", 0.0),
            "signature": ", ".join(sigs) if sigs else "unknown",
        })

    # High-delta NO_BUG seeds: delta > 0.5 × min bug delta
    if bug_entries:
        min_bug_delta = min(e.get("delta", 0.0) for e in bug_entries)
        threshold = 0.5 * min_bug_delta
        for e in no_bug_entries:
            if e.get("delta", 0.0) > threshold:
                kb["high_delta_seeds"].append({
                    "c": e.get("c", {}),
                    "w": e.get("w", {}),
                    "delta": e.get("delta", 0.0),
                })

    # Signature priors: fraction of bugs per signature
    all_sig_counts: dict[str, int] = {}
    for e in bug_entries:
        for sig in e.get("signatures", []):
            all_sig_counts[sig] = all_sig_counts.get(sig, 0) + 1
    total = sum(all_sig_counts.values())
    if total > 0:
        kb["signature_priors"] = {s: c / total for s, c in all_sig_counts.items()}

    # Merge with existing KB (append seeds, update priors)
    if existing_kb:
        merged = copy.deepcopy(existing_kb)
        existing_seeds = {(json.dumps(s["c"], sort_keys=True), json.dumps(s["w"], sort_keys=True))
                          for s in merged.get("bug_seeds", [])}
        for s in kb["bug_seeds"]:
            key = (json.dumps(s["c"], sort_keys=True), json.dumps(s["w"], sort_keys=True))
            if key not in existing_seeds:
                merged["bug_seeds"].append(s)
        existing_hd = {(json.dumps(s["c"], sort_keys=True), json.dumps(s["w"], sort_keys=True))
                       for s in merged.get("high_delta_seeds", [])}
        for s in kb["high_delta_seeds"]:
            key = (json.dumps(s["c"], sort_keys=True), json.dumps(s["w"], sort_keys=True))
            if key not in existing_hd:
                merged["high_delta_seeds"].append(s)
        if kb["signature_priors"]:
            merged["signature_priors"] = kb["signature_priors"]
        return merged

    return kb
