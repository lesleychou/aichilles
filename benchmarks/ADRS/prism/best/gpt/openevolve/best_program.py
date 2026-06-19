GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR via bisection on threshold T using a greedy feasibility check.
    KVPR_i = sum(req_rate/slo) / (GPU_MEM_SIZE - sum(model_size)) on GPU i.
    """
    M = GPU_MEM_SIZE
    items = [(m, m.req_rate / m.slo, m.model_size) for m in models]
    for _, _, s in items:
        if s >= M:
            raise ValueError(f"Model size {s} GB too large for GPU memory {M} GB")

    # Lower bound on achievable KVPR: at least max_i a_i/(M - s_i)
    LB = 0.0
    for _, a, s in items:
        t = a / (M - s)
        if t > LB:
            LB = t

    def feasible(T):
        # Best-fit under transformed slack: require T*rem - w >= a + T*s and rem - s > 0
        order = sorted(items, key=lambda x: (x[1] + T * x[2]), reverse=True)
        rem = [M] * gpu_num
        w = [0.0] * gpu_num
        pl = {i: [] for i in range(gpu_num)}
        for m, a, s in order:
            demand = a + T * s
            best = None
            best_residual = float('inf')
            best_mnew = -1.0
            for i in range(gpu_num):
                m_new = rem[i] - s
                if m_new > 0:
                    slack = T * rem[i] - w[i]
                    if slack >= demand:
                        residual = slack - demand
                        if residual < best_residual or (residual == best_residual and m_new > best_mnew):
                            best_residual = residual
                            best_mnew = m_new
                            best = i
            if best is None:
                return None
            pl[best].append(m)
            w[best] += a
            rem[best] -= s
        return pl

    # Expand T until feasible, then refine with binary search
    T = LB if LB > 1.0 else 1.0
    placement = feasible(T)
    tries = 0
    while placement is None and tries < 32:
        T *= 2.0
        placement = feasible(T)
        tries += 1

    if placement is None:
        # Fallback to robust greedy minimizing post-placement global max KVPR
        placement = {i: [] for i in range(gpu_num)}
        rem = [M] * gpu_num
        w = [0.0] * gpu_num
        order = sorted(items, key=lambda x: (x[1], x[2]), reverse=True)
        for m, a, s in order:
            best = None
            best_global = float('inf')
            cur = [(w[i] / rem[i]) if rem[i] > 0 else float('inf') for i in range(gpu_num)]
            for i in range(gpu_num):
                m_new = rem[i] - s
                if m_new > 0:
                    k_local = (w[i] + a) / m_new
                    max_other = 0.0
                    for j in range(gpu_num):
                        if j != i and cur[j] > max_other:
                            max_other = cur[j]
                    g = k_local if k_local > max_other else max_other
                    if g < best_global:
                        best_global = g
                        best = i
            if best is None:
                raise ValueError(
                    f"Unable to place model of size {s} GB on any GPU. Remaining per-GPU memory: {rem}"
                )
            placement[best].append(m)
            w[best] += a
            rem[best] -= s
        return placement

    lo, hi = LB, T
    best_pl = placement
    for _ in range(24):
        mid = (lo + hi) / 2.0
        pl = feasible(mid)
        if pl is None:
            lo = mid
        else:
            best_pl = pl
            hi = mid
    return best_pl

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
