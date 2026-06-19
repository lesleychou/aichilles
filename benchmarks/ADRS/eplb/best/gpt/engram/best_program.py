# EPLB Optimizer: Try perturbation with different seed as 5th candidate
# Base 4: cap=0, cap=4, cap=5, pert(42)_cap=4
# 5th: pert(7)_cap=5

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group].item()
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int,
        max_cnt: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    logcnt_f = logcnt.float()
    weight_f = weight.float()
    
    for i in range(num_log, num_phy):
        ratios = weight_f / logcnt_f
        if max_cnt > 0:
            ratios = ratios.clone()
            ratios[logcnt >= max_cnt] = -1.0
        redundant_indices = ratios.max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
        logcnt_f[arangen, redundant_indices] += 1.0
    return phy2log, rank, logcnt


def rebalance_experts_global(weight, num_physical_experts, num_gpus, max_cnt=0):
    num_layers, num_logical_experts = weight.shape
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm):
        inv = torch.empty_like(perm)
        inv.scatter_(1, perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape))
        return inv

    phy2log, phyrank, logcnt = replicate_experts(weight, num_physical_experts, max_cnt)
    tokens_per_phy = (weight / logcnt).gather(-1, phy2log)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2log = phy2log.gather(-1, pphy2phy)
    pphyrank = phyrank.gather(-1, pphy2phy)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    phy_per_gpu = num_replicas // num_gpus
    
    def get_max_loads(p2l, lc):
        ew = weight / lc.float()
        gl = ew.gather(-1, p2l).view(num_layers, num_gpus, phy_per_gpu).sum(-1)
        return gl.max(-1).values
    
    candidates = []
    
    for cap in [0, 4, 5]:
        p2l, pr, lc = rebalance_experts_global(weight, num_replicas, num_gpus, max_cnt=cap)
        candidates.append((p2l, pr, lc, get_max_loads(p2l, lc)))
    
    mean_w = weight.abs().mean()
    
    # Perturbation 1: seed=42, cap=4
    torch.manual_seed(42)
    noise = torch.randn_like(weight) * mean_w * 0.001
    perturbed = (weight + noise).clamp(min=0)
    p2l, pr, lc = rebalance_experts_global(perturbed, num_replicas, num_gpus, max_cnt=4)
    candidates.append((p2l, pr, lc, get_max_loads(p2l, lc)))
    
    # Perturbation 2: seed=7, cap=5
    torch.manual_seed(7)
    noise = torch.randn_like(weight) * mean_w * 0.001
    perturbed = (weight + noise).clamp(min=0)
    p2l, pr, lc = rebalance_experts_global(perturbed, num_replicas, num_gpus, max_cnt=5)
    candidates.append((p2l, pr, lc, get_max_loads(p2l, lc)))
    
    # Per-layer best selection
    all_max = torch.stack([c[3] for c in candidates])
    best_idx = all_max.argmin(dim=0)
    
    phy2log = candidates[0][0].clone()
    phyrank = candidates[0][1].clone()
    logcnt = candidates[0][2].clone()
    
    for layer in range(num_layers):
        idx = best_idx[layer].item()
        if idx != 0:
            phy2log[layer] = candidates[idx][0][layer]
            phyrank[layer] = candidates[idx][1][layer]
            logcnt[layer] = candidates[idx][2][layer]
    
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1, dtype=torch.int64, device=logcnt.device)
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=log2phy.device).expand(num_layers, -1))
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts"]