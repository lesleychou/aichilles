GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Exact minimax KVPR via binary search with branch-and-bound feasibility.
    - Transform to effective sizes e_i(k) = s_i + w_i / k where s_i=model_size, w_i=req_rate/slo.
      If items e_i fit into gpu_num bins of capacity 80, then per-GPU KVPR <= k.
    - Use branch-and-bound to exactly decide feasibility for a given k:
        * Items ordered by descending e_i (tie-break by s_i).
        * Try bins in best-fit order (least remaining effective capacity first).
        * Strong pruning: global capacity lower bound, symmetry skipping by identical remainders,
          and memoization by (idx, sorted rounded remainders).
        * Node limit with graceful fallback to a fast BFD heuristic to ensure timely completion.
    - Binary search the smallest feasible k and return the corresponding placement.
    """
    eps = 1e-12
    S = [m.model_size for m in models]
    W = [m.req_rate / (m.slo if m.slo > 0 else eps) for m in models]
    n = len(models)

    if any(s > GPU_MEM_SIZE for s in S):
        raise ValueError("A model exceeds GPU memory capacity.")
    total_s = sum(S)
    total_w = sum(W)
    if total_s > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds aggregate GPU memory.")

    # Memory-only quick path
    if total_w == 0:
        placement = {i: [] for i in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        order = sorted(range(n), key=lambda i: S[i], reverse=True)
        for idx in order:
            s = S[idx]
            best, best_res = -1, float('inf')
            for g in range(gpu_num):
                res = rem[g] - s
                if res >= -1e-12 and res < best_res:
                    best, best_res = g, res
            if best == -1:
                raise ValueError("Unable to place models by memory.")
            placement[best].append(models[idx])
            rem[best] -= s
        return placement

    # Heuristic BFD packer used as fallback and for fast upper-bound seeding
    def bfd_pack(k, return_place=True):
        e = [S[i] + W[i] / max(k, eps) for i in range(n)]
        if max(e) > GPU_MEM_SIZE + 1e-9:
            return None
        order = sorted(range(n), key=lambda i: (e[i], S[i]), reverse=True)
        eff_rem = [GPU_MEM_SIZE] * gpu_num
        place = [[] for _ in range(gpu_num)]
        for idx in order:
            best, best_gap = -1, float('inf')
            for g in range(gpu_num):
                gap = eff_rem[g] - e[idx]
                if gap >= -1e-9 and gap < best_gap - 1e-12:
                    best, best_gap = g, gap
            if best == -1:
                return None
            eff_rem[best] -= e[idx]
            if return_place:
                place[best].append(models[idx])
        if return_place:
            return {i: place[i] for i in range(gpu_num)}
        return True

    # Exact feasibility via branch-and-bound; returns (True/False/None, placement_or_None)
    # None status indicates node limit exceeded -> caller may fallback.
    def exact_pack(k, need_assignment=False, node_limit=100000):
        e = [S[i] + W[i] / max(k, eps) for i in range(n)]
        if max(e) > GPU_MEM_SIZE + 1e-9:
            return (False, None)
        order = sorted(range(n), key=lambda i: (e[i], S[i]), reverse=True)
        eff_rem = [GPU_MEM_SIZE] * gpu_num
        assignment = [-1] * n if need_assignment else None
        total_e_remaining = sum(e[i] for i in order)
        nodes = [0]
        memo = {}
        round_digits = 6

        def key(idx):
            # bins are unlabeled: sort remainders for canonical state
            return (idx, tuple(sorted(round(r, round_digits) for r in eff_rem)))

        def dfs(idx, rem_e):
            # Node limit guard
            nodes[0] += 1
            if nodes[0] > node_limit:
                return None  # signal limit
            # Global capacity bound
            if rem_e > sum(eff_rem) + 1e-9:
                return False
            if idx == len(order):
                return True
            kstate = key(idx)
            if kstate in memo:
                return memo[kstate]
            item = order[idx]
            seen = set()
            # Best-fit: try bins with least remaining effective capacity first
            candidates = sorted(range(gpu_num), key=lambda g: eff_rem[g])
            placed_any = False
            for g in candidates:
                rem = eff_rem[g]
                sig = round(rem, round_digits)
                if sig in seen:
                    continue
                if e[item] <= rem + 1e-9:
                    seen.add(sig)
                    eff_rem[g] = rem - e[item]
                    if need_assignment:
                        assignment[item] = g
                    res = dfs(idx + 1, rem_e - e[item])
                    if res is None:
                        # propagate node limit hit
                        if need_assignment:
                            assignment[item] = -1
                        eff_rem[g] = rem
                        return None
                    if res:
                        memo[kstate] = True
                        return True
                    if need_assignment:
                        assignment[item] = -1
                    eff_rem[g] = rem
                    placed_any = True
            if not placed_any:
                memo[kstate] = False
                return False
            memo[kstate] = False
            return False

        result = dfs(0, total_e_remaining)
        if result is None:
            return (None, None)
        if result:
            if need_assignment:
                placement = {i: [] for i in range(gpu_num)}
                for i in range(n):
                    g = assignment[i]
                    if g < 0:
                        # Should not happen, but guard anyway
                        return (False, None)
                    placement[g].append(models[i])
                return (True, placement)
            return (True, None)
        return (False, None)

    # Lower bound on k
    lb = eps
    for s, w in zip(S, W):
        den = GPU_MEM_SIZE - s
        if den <= 0:
            raise ValueError("Model too large for GPU memory.")
        lb = max(lb, w / den)
    den_tot = gpu_num * GPU_MEM_SIZE - total_s
    if den_tot > 0:
        lb = max(lb, total_w / den_tot)

    # Find an initial feasible upper bound quickly using BFD; double if needed
    ub = max(lb * 2.0, 1.0)
    guard = 0
    placement_for_ub = bfd_pack(ub, return_place=True)
    while placement_for_ub is None and guard < 60:
        ub *= 2.0
        placement_for_ub = bfd_pack(ub, return_place=True)
        guard += 1
    if placement_for_ub is None:
        # As a last resort, greedy memory best-fit
        placement = {i: [] for i in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        order = sorted(range(n), key=lambda i: S[i], reverse=True)
        for idx in order:
            s = S[idx]
            best, best_res = -1, float('inf')
            for g in range(gpu_num):
                res = rem[g] - s
                if res >= -1e-12 and res < best_res:
                    best, best_res = g, res
            if best == -1:
                raise ValueError("Unable to place models by memory.")
            placement[best].append(models[idx])
            rem[best] -= s
        return placement

    # Tighten ub using exact feasibility if cheap; keep fallback placement if exact hits limit
    ok_exact, _ = exact_pack(ub, need_assignment=False)
    if ok_exact is False:
        # Should not happen (BFD found a feasible pack), but be safe
        ok_exact = True

    # Binary search for minimal feasible k using exact feasibility with node limit
    best_k = ub
    exact_failed = False
    for _ in range(35):
        mid = 0.5 * (lb + ub)
        status, _ = exact_pack(mid, need_assignment=False)
        if status is True:
            best_k = mid
            ub = mid
        elif status is False:
            lb = mid
        else:
            # Node limit exceeded, break and fallback later
            exact_failed = True
            break
        if ub - lb <= max(1e-4, 1e-4 * ub):
            break

    # Reconstruct placement for best_k using exact method; fallback to BFD if needed
    status, placement = exact_pack(best_k, need_assignment=True)
    if status is True and placement is not None:
        return placement
    # Fallbacks: try a slightly relaxed k with exact, then BFD
    for factor in (1.01, 1.03, 1.05):
        status, placement = exact_pack(best_k * factor, need_assignment=True)
        if status is True and placement is not None:
            return placement
    place_bfd = bfd_pack(best_k, return_place=True)
    if place_bfd is not None:
        return place_bfd
    # Last resort: use the feasible placement we found when seeding ub
    return placement_for_ub

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Test the algorithm

    from evaluator import generate_test_gpu_models
    from evaluator import calculate_kvcache_pressure
    from evaluator import safe_float
    import numpy as np

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):

        results = compute_model_placement(gpu_num, gpu_models)
        max_kvpr = calculate_kvcache_pressure(results)
        all_kvpr.append(safe_float(max_kvpr))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr


    print(f"Max KVPR: {avg_kvpr:.3f}")
