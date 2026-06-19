"""
Fixed per-app run_workload for EPLB.

Workload params (all JSON-serializable — no tensors):
  seed         int   random seed for weight generation (ensures P and P' see identical tensors)
  n_layers     int   number of MoE layers
  n_experts    int   number of logical experts (must be multiple of num_groups)
  load_family  str   "uniform" | "sparse_hot" | "bimodal" | "alternating"
  skew         float load concentration factor (higher = more skewed)
  hot_fraction float fraction of hot experts in sparse_hot (0.0–1.0)
  num_replicas int   physical expert replicas (default 288)
  num_groups   int   expert groups (default 8)
  num_nodes    int   number of nodes (default 4)
  num_gpus     int   number of GPUs (default 32)

Returns a dict of scalar quality metrics for the optimality oracle.
"""
import math
import torch


def _nearest_divisor(n: int, value: int) -> int:
    """Return the divisor of n closest to value."""
    value = max(1, value)
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    return min(divisors, key=lambda d: abs(d - value))


def _make_weight(n_layers: int, n_experts: int, load_family: str,
                 skew: float, hot_fraction: float) -> torch.Tensor:
    """Synthesize an expert-load tensor [n_layers, n_experts]."""
    if load_family == "uniform":
        return torch.rand(n_layers, n_experts)

    elif load_family == "sparse_hot":
        # A small fraction of experts receive most of the load.
        weight = torch.rand(n_layers, n_experts) * 0.1
        n_hot = max(1, int(n_experts * hot_fraction))
        for layer in range(n_layers):
            hot_idx = torch.randperm(n_experts)[:n_hot]
            weight[layer, hot_idx] += skew
        return weight

    elif load_family == "bimodal":
        # Two groups with clearly different load levels.
        weight = torch.zeros(n_layers, n_experts)
        half = n_experts // 2
        weight[:, :half] = torch.rand(n_layers, half) * skew
        weight[:, half:] = torch.rand(n_layers, n_experts - half)
        return weight

    elif load_family == "alternating":
        # Odd/even experts alternate between high and low load.
        base = torch.rand(n_layers, n_experts)
        mask = (torch.arange(n_experts) % 2 == 0).float()
        return base + mask.unsqueeze(0) * skew

    else:
        return torch.rand(n_layers, n_experts)


def run_workload(program_module, workload: dict):
    seed         = int(workload.get("seed",          42))
    n_layers     = int(workload.get("n_layers",       8))
    n_experts    = int(workload.get("n_experts",     64))
    load_family  = str(workload.get("load_family",  "uniform"))
    skew         = float(workload.get("skew",         1.0))
    hot_fraction = float(workload.get("hot_fraction", 0.05))
    num_replicas = int(workload.get("num_replicas",  288))
    num_groups   = int(workload.get("num_groups",      8))
    num_nodes    = int(workload.get("num_nodes",       4))
    num_gpus     = int(workload.get("num_gpus",       32))

    # Snap config params to values that satisfy rebalance_experts constraints.
    num_groups   = _nearest_divisor(n_experts, num_groups)
    num_nodes    = _nearest_divisor(num_groups, num_nodes)
    num_gpus     = max(num_nodes, math.ceil(num_gpus / num_nodes) * num_nodes)
    num_replicas = math.ceil(max(num_replicas, n_experts) / num_gpus) * num_gpus

    # Seed torch before weight generation so P and P' receive identical tensors.
    torch.manual_seed(seed)
    weight = _make_weight(n_layers, n_experts, load_family, skew, hot_fraction)

    _phy2log, log2phy, logcnt = program_module.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )

    # Quality metrics for the optimality oracle — must be a dict of scalars.
    import evaluator as _ev
    score_gpu, score_expert = _ev.simulate_inference(log2phy, logcnt, weight)
    return {
        "balancedness_score_gpu":    float(score_gpu),
        "balancedness_score_expert": float(score_expert),
    }
