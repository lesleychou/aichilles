import json as _json
import random as _random

from txn_simulator import Workload


def _generate_txn_workload(seed, num_txns, num_resources, op_length, write_ratio, hot_resources):
    """
    Generate a transaction workload JSON string deterministically from seed.

    hot_resources: number of resources that appear 5x more often than the rest (0 = uniform).
    write_ratio:   integer 0-100, percentage of operations that are writes.
    """
    rng = _random.Random(seed)
    all_resources = list(range(1, num_resources + 1))
    if hot_resources > 0 and hot_resources < num_resources:
        hot_pool = all_resources[:hot_resources] * 5 + all_resources[hot_resources:]
    else:
        hot_pool = all_resources

    txns = {}
    for i in range(num_txns):
        ops = []
        for _ in range(op_length):
            resource = rng.choice(hot_pool)
            op_type = "w" if rng.randint(1, 100) <= write_ratio else "r"
            ops.append(f"{op_type}-{resource}")
        txns[f"txn{i}"] = " ".join(ops)

    return _json.dumps(txns)


def run_workload(program_module, workload: dict):
    seed = int(workload.get("seed", 42))
    # Coverage-traced execution (harness) is ~3x slower than bare execution.
    # Cap num_txns at 12 and num_seqs at 10 so coverage runs finish within the
    # 30s harness timeout even on demanding workloads (worst case ~16s measured).
    # Clamp to grammar minimums: LLM-generated generate_workload.py may violate them.
    num_txns = max(5, min(int(workload.get("num_txns", 10)), 12))
    num_resources = max(5, int(workload.get("num_resources", 20)))
    op_length = max(2, int(workload.get("op_length", 8)))
    write_ratio = max(0, min(int(workload.get("write_ratio", 50)), 100))
    hot_resources = max(0, min(int(workload.get("hot_resources", 0)), num_resources - 1))
    num_seqs = max(1, min(int(workload.get("num_seqs", 10)), 10))

    workload_json = _generate_txn_workload(
        seed, num_txns, num_resources, op_length, write_ratio, hot_resources
    )
    wl = Workload(workload_json)

    # Seed algorithm's random calls with the same seed so P and P' see identical draws
    _random.seed(seed)
    makespan, schedule = program_module.get_best_schedule(wl, num_seqs)
    combined_score = 1000.0 / (1.0 + makespan) * 1000.0
    return {"combined_score": combined_score}
