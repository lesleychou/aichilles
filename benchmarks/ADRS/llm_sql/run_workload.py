import os

import pandas as pd


def run_workload(program_module, workload: dict):
    dataset = workload["dataset"]
    col_merge = workload.get("col_merge", [])

    # Resolve dataset path relative to the llm_sql datasets/ directory.
    # Walk up from program_module's location to find the app root (contains evaluator.py).
    if not os.path.isabs(dataset):
        _d = os.path.dirname(os.path.abspath(program_module.__file__))
        while _d != os.path.dirname(_d):
            if os.path.exists(os.path.join(_d, "evaluator.py")):
                break
            _d = os.path.dirname(_d)
        dataset = os.path.join(_d, "datasets", os.path.basename(dataset))

    master_df = pd.read_csv(dataset, dtype=str).fillna("")
    reordered, meta = program_module.Evolved().reorder(
        master_df,
        early_stop=workload.get("early_stop", 100000),
        distinct_value_threshold=workload.get("distinct_value_threshold", 0.7),
        row_stop=workload.get("row_stop", 4),
        col_stop=workload.get("col_stop", 2),
        col_merge=col_merge,
    )

    from utils import evaluate_df_prefix_hit_cnt
    results = evaluate_df_prefix_hit_cnt(reordered)
    # results is (total_count, hit_rate_pct); hit_rate_pct is 0-100, higher is better
    hit_rate = float(results[1]) / 100.0
    return {"hit_rate": hit_rate}
