# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
"""

# EVOLVE-BLOCK-START

import heapq
import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Greedy lightest-pack with capacity per row:
    - Sort items desc by weight.
    - Assign next heaviest to the current lightest pack with remaining capacity.
    Ensures n/num_packs items per pack and balances total weight.
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    dev = weight.device
    sorted_idx = weight.float().argsort(-1, descending=True)
    pack_weights = torch.zeros(num_layers, num_packs, dtype=weight.dtype, device=dev)
    pack_counts = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=dev)
    pack_index = torch.full((num_layers, num_groups), -1, dtype=torch.int64, device=dev)
    rank_in_pack = torch.full_like(pack_index, -1)
    rows = torch.arange(num_layers, device=dev, dtype=torch.int64)
    cap = groups_per_pack
    inf = torch.tensor(float("inf"), device=dev, dtype=pack_weights.dtype)

    for t in range(num_groups):
        idx = sorted_idx[:, t]
        free = pack_counts < cap
        masked = torch.where(free, pack_weights, inf)
        chosen_pack = masked.argmin(dim=1)
        ranks = pack_counts.gather(1, chosen_pack.unsqueeze(1)).squeeze(1)

        pack_index[rows, idx] = chosen_pack
        rank_in_pack[rows, idx] = ranks

        w_sel = weight[rows, idx].to(pack_weights.dtype)
        pack_weights[rows, chosen_pack] += w_sel
        pack_counts[rows, chosen_pack] += 1

    return pack_index, rank_in_pack


def _waterfill_counts(loads: torch.Tensor, m: int) -> torch.Tensor:
    """
    Water-filling replica counts via threshold search:
    - Find tau s.t. sum_e max(ceil(load_e / tau), 1) == m using binary search.
    - Clamp min count to 1. Adjust to match m exactly by greedy increase/decrease
      using marginal change in per-replica load.
    Args:
        loads: [num_logical], per-expert load for one layer (float tensor)
        m: number of physical experts (replicas)
    Returns:
        counts: [num_logical], replicas per logical expert (int64)
    """
    n = loads.numel()
    assert m >= n, "num_replicas must be >= num_logical_experts"
    if n == 0:
        return torch.zeros(0, dtype=torch.int64, device=loads.device)
    ones = torch.ones(n, dtype=torch.int64, device=loads.device)
    maxw = loads.max()
    # All-zero load: start with 1 per expert, distribute extras arbitrarily
    if maxw.item() == 0.0:
        counts = ones.clone()
        extra = m - n
        if extra > 0:
            # Prefer deterministic: allocate extras to lowest indices
            idx = torch.arange(n, device=loads.device)
            counts[idx[:extra]] += 1
        return counts
    # Binary search for tau on (0, maxw]
    lo = 0.0
    hi = maxw.item()
    for _ in range(24):
        mid = (lo + hi) / 2.0
        denom = max(mid, 1e-12)
        tentative = torch.ceil(loads / denom).to(torch.int64)
        tentative = torch.maximum(tentative, ones)
        s = int(tentative.sum().item())
        if s > m:
            lo = mid
        else:
            hi = mid
    denom = max(hi, 1e-12)
    counts = torch.ceil(loads / denom).to(torch.int64)
    counts = torch.maximum(counts, ones)
    total = int(counts.sum().item())
    if total > m:
        need = total - m
        # Decrease counts with smallest harm: Δ = w/(k-1) - w/k
        mask = counts > 1
        harm = torch.full_like(loads, float("inf"))
        harm[mask] = loads[mask] / (counts[mask] - 1) - loads[mask] / counts[mask]
        # Select 'need' smallest harm
        k = min(need, int(mask.sum().item()))
        if k > 0:
            sel = torch.topk(-harm, k, largest=True).indices  # negate to pick smallest harm
            counts[sel] -= 1
    elif total < m:
        need = m - total
        # Increase counts where improvement is largest: Δ = w/k - w/(k+1)
        improve = loads / counts - loads / (counts + 1)
        k = min(need, n)
        if k > 0:
            sel = torch.topk(improve, k, largest=True).indices
            counts[sel] += 1
    return counts


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Per-layer makespan heuristic:
      - Water-fill counts so sum replicas = num_physical_experts (min 1 each).
      - Per-replica share = load/count.
      - Capacity-constrained LPT placement using a min-heap over GPUs.
    Avoids large replica tensors and reduces runtime while keeping GPU and expert balance.
    Returns:
        physical_to_logical_map [layers, m], replica_rank [layers, m], logical_count [layers, E]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_physical_experts % num_gpus == 0
    phy_per_gpu = num_physical_experts // num_gpus
    device = weight.device

    phy2log = torch.full((num_layers, num_physical_experts),
                         -1, dtype=torch.int64, device=device)
    phyrank = torch.zeros((num_layers, num_physical_experts),
                          dtype=torch.int64, device=device)
    logcnt = torch.zeros((num_layers, num_logical_experts),
                         dtype=torch.int64, device=device)

    for l in range(num_layers):
        loads = weight[l].float()
        counts = _waterfill_counts(loads, num_physical_experts)
        logcnt[l] = counts

        shares = torch.where(counts > 0,
                             loads / counts.to(loads.dtype),
                             torch.zeros_like(loads))
        order = torch.argsort(shares, descending=True).tolist()

        heap = [(0.0, g) for g in range(num_gpus)]
        heapq.heapify(heap)
        next_slot = [0] * num_gpus

        assigned = 0
        for e in order:
            k = int(counts[e].item())
            if k <= 0:
                continue
            share = float(shares[e].item())
            for r in range(k):
                cur, g = heapq.heappop(heap)
                pos = g * phy_per_gpu + next_slot[g]
                phy2log[l, pos] = e
                phyrank[l, pos] = r
                next_slot[g] += 1
                assigned += 1
                updated = cur + share
                if next_slot[g] < phy_per_gpu:
                    heapq.heappush(heap, (updated, g))
        # Safety: fill any remaining slots deterministically (should rarely trigger)
        if assigned < num_physical_experts:
            for g in range(num_gpus):
                while next_slot[g] < phy_per_gpu and assigned < num_physical_experts:
                    pos = g * phy_per_gpu + next_slot[g]
                    phy2log[l, pos] = 0
                    phyrank[l, pos] = 0
                    next_slot[g] += 1
                    assigned += 1

    return phy2log, phyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point: per-layer makespan minimization with water-filling counts and
    GPU-capacity constrained packing. Fast O(L*(E + R) + L*R) heuristic with
    strong balance in expert load and per-GPU load.
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float()
    # Directly use the makespan-optimized hierarchical stub (topology-agnostic counts)
    phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
        weight, num_replicas, num_groups, num_nodes, num_gpus)
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=log2phy.device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

