GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR using ILP solver (PuLP/CBC) with binary search on target T.
    
    For target T, feasibility: can we assign models to GPUs so that for each GPU g:
      sum_i (rate_i + T*size_i) * x_{ig} <= T * GPU_MEM
    This is a standard bin packing ILP. Binary search on T to find min feasible T.
    
    Falls back to heuristic bin packing + local search if ILP is too slow.
    """
    import pulp
    import time
    
    rates = [m.req_rate / m.slo for m in models]
    sizes = [m.model_size for m in models]
    n = len(models)
    
    def compute_max_kvpr(assignment):
        w = [0.0] * gpu_num
        s = [0.0] * gpu_num
        for i, g in enumerate(assignment):
            w[g] += rates[i]
            s[g] += sizes[i]
        mx = 0.0
        for g in range(gpu_num):
            rem = GPU_MEM_SIZE - s[g]
            if rem <= 0:
                return float('inf')
            kvpr = w[g] / rem
            if kvpr > mx:
                mx = kvpr
        return mx
    
    def solve_ilp(T, time_limit=3, warm_start=None):
        """Check if assignment with max KVPR <= T is feasible using ILP."""
        capacity = T * GPU_MEM_SIZE
        weights = [rates[i] + T * sizes[i] for i in range(n)]
        
        prob = pulp.LpProblem("KVPR", pulp.LpMinimize)
        
        x = [[pulp.LpVariable(f"x_{i}_{g}", cat='Binary') for g in range(gpu_num)] for i in range(n)]
        
        # Minimize slack (helps find tighter solutions)
        # Actually just use 0 for feasibility
        prob += 0
        
        for i in range(n):
            prob += pulp.lpSum(x[i][g] for g in range(gpu_num)) == 1
        
        for g in range(gpu_num):
            prob += pulp.lpSum(weights[i] * x[i][g] for i in range(n)) <= capacity
        
        for g in range(gpu_num):
            prob += pulp.lpSum(sizes[i] * x[i][g] for i in range(n)) <= GPU_MEM_SIZE - 0.001
        
        # Symmetry breaking
        for g in range(gpu_num - 1):
            prob += pulp.lpSum(weights[i] * x[i][g] for i in range(n)) >= \
                    pulp.lpSum(weights[i] * x[i][g+1] for i in range(n))
        
        # Warm start
        if warm_start is not None:
            for i in range(n):
                for g in range(gpu_num):
                    x[i][g].setInitialValue(1 if warm_start[i] == g else 0)
        
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, warmStart=(warm_start is not None))
        prob.solve(solver)
        
        if prob.status == 1:
            assignment = [0] * n
            for i in range(n):
                for g in range(gpu_num):
                    val = pulp.value(x[i][g])
                    if val is not None and val > 0.5:
                        assignment[i] = g
                        break
            return assignment
        return None
    
    def try_pack(T, mode='best'):
        capacity = T * GPU_MEM_SIZE
        weights = [rates[i] + T * sizes[i] for i in range(n)]
        order = sorted(range(n), key=lambda i: weights[i], reverse=True)
        bl = [0.0] * gpu_num
        bm = [0.0] * gpu_num
        asgn = [0] * n
        for i in order:
            wt = weights[i]
            si = sizes[i]
            bg = -1
            if mode == 'best':
                bv = float('inf')
                for g in range(gpu_num):
                    if bl[g] + wt <= capacity + 1e-9 and bm[g] + si < GPU_MEM_SIZE:
                        r = capacity - bl[g] - wt
                        if r < bv:
                            bv = r
                            bg = g
            elif mode == 'worst':
                bv = -1.0
                for g in range(gpu_num):
                    if bl[g] + wt <= capacity + 1e-9 and bm[g] + si < GPU_MEM_SIZE:
                        r = capacity - bl[g] - wt
                        if r > bv:
                            bv = r
                            bg = g
            else:
                for g in range(gpu_num):
                    if bl[g] + wt <= capacity + 1e-9 and bm[g] + si < GPU_MEM_SIZE:
                        bg = g
                        break
            if bg == -1:
                return None
            asgn[i] = bg
            bl[bg] += wt
            bm[bg] += si
        return asgn
    
    def greedy_min_kvpr(order):
        wg = [0.0] * gpu_num
        sg = [0.0] * gpu_num
        asgn = [0] * n
        for i in order:
            ri, si = rates[i], sizes[i]
            bg = -1
            bmax = float('inf')
            for g in range(gpu_num):
                if sg[g] + si >= GPU_MEM_SIZE:
                    continue
                rem = GPU_MEM_SIZE - sg[g] - si
                nk = (wg[g] + ri) / rem
                cm = nk
                for g2 in range(gpu_num):
                    if g2 != g and wg[g2] > 0:
                        k = wg[g2] / (GPU_MEM_SIZE - sg[g2])
                        if k > cm:
                            cm = k
                if cm < bmax:
                    bmax = cm
                    bg = g
            if bg == -1:
                bg = min(range(gpu_num), key=lambda gg: sg[gg])
            asgn[i] = bg
            wg[bg] += ri
            sg[bg] += si
        return asgn
    
    def refine(assignment):
        w = [0.0] * gpu_num
        s = [0.0] * gpu_num
        gm = [[] for _ in range(gpu_num)]
        for i, g in enumerate(assignment):
            w[g] += rates[i]
            s[g] += sizes[i]
            gm[g].append(i)
        
        kc = [0.0] * gpu_num
        for g in range(gpu_num):
            rem = GPU_MEM_SIZE - s[g]
            kc[g] = w[g] / rem if rem > 0 else float('inf')
        
        for iteration in range(500):
            mx = max(kc)
            wg_idx = kc.index(mx)
            if mx <= 0:
                break
            
            bi = None
            bv = mx
            for il, i in enumerate(gm[wg_idx]):
                ri, si = rates[i], sizes[i]
                for tg in range(gpu_num):
                    if tg == wg_idx or s[tg] + si >= GPU_MEM_SIZE:
                        continue
                    sk = (w[wg_idx] - ri) / (GPU_MEM_SIZE - s[wg_idx] + si) if w[wg_idx] > ri else 0
                    tk = (w[tg] + ri) / (GPU_MEM_SIZE - s[tg] - si)
                    nm = max(sk, tk)
                    for g in range(gpu_num):
                        if g != wg_idx and g != tg and kc[g] > nm:
                            nm = kc[g]
                    if nm < bv - 1e-12:
                        bv = nm
                        bi = ('m', il, i, tg)
            
            if bi is None:
                bs = None
                bsv = mx
                for ia, i in enumerate(gm[wg_idx]):
                    ri, si = rates[i], sizes[i]
                    for tg in range(gpu_num):
                        if tg == wg_idx:
                            continue
                        for ib, j in enumerate(gm[tg]):
                            rj, sj = rates[j], sizes[j]
                            rw = GPU_MEM_SIZE - s[wg_idx] + si - sj
                            rt = GPU_MEM_SIZE - s[tg] + sj - si
                            if rw <= 0 or rt <= 0:
                                continue
                            kww = (w[wg_idx] - ri + rj) / rw if w[wg_idx] - ri + rj > 0 else 0
                            kwt = (w[tg] - rj + ri) / rt
                            nm = max(kww, kwt)
                            for g in range(gpu_num):
                                if g != wg_idx and g != tg and kc[g] > nm:
                                    nm = kc[g]
                            if nm < bsv - 1e-12:
                                bsv = nm
                                bs = (ia, i, tg, ib, j)
                if bs is None:
                    break
                ia, i, tg, ib, j = bs
                ri, si = rates[i], sizes[i]
                rj, sj = rates[j], sizes[j]
                gm[wg_idx][ia] = j
                gm[tg][ib] = i
                w[wg_idx] += -ri + rj; s[wg_idx] += -si + sj
                w[tg] += -rj + ri; s[tg] += -sj + si
                kc[wg_idx] = w[wg_idx] / (GPU_MEM_SIZE - s[wg_idx]) if s[wg_idx] < GPU_MEM_SIZE else float('inf')
                kc[tg] = w[tg] / (GPU_MEM_SIZE - s[tg]) if s[tg] < GPU_MEM_SIZE else float('inf')
            else:
                _, il, i, tg = bi
                ri, si = rates[i], sizes[i]
                gm[wg_idx].pop(il)
                gm[tg].append(i)
                w[wg_idx] -= ri; s[wg_idx] -= si
                w[tg] += ri; s[tg] += si
                kc[wg_idx] = w[wg_idx] / (GPU_MEM_SIZE - s[wg_idx]) if s[wg_idx] < GPU_MEM_SIZE else float('inf')
                kc[tg] = w[tg] / (GPU_MEM_SIZE - s[tg]) if s[tg] < GPU_MEM_SIZE else float('inf')
        
        # All-pairs swaps
        for iteration in range(200):
            mx = max(kc)
            bs = None
            bsv = mx
            for g1 in range(gpu_num):
                for ia, i in enumerate(gm[g1]):
                    ri, si = rates[i], sizes[i]
                    for g2 in range(g1 + 1, gpu_num):
                        for ib, j in enumerate(gm[g2]):
                            rj, sj = rates[j], sizes[j]
                            r1 = GPU_MEM_SIZE - s[g1] + si - sj
                            r2 = GPU_MEM_SIZE - s[g2] + sj - si
                            if r1 <= 0 or r2 <= 0:
                                continue
                            k1 = (w[g1] - ri + rj) / r1 if w[g1] - ri + rj > 0 else 0
                            k2 = (w[g2] - rj + ri) / r2 if w[g2] - rj + ri > 0 else 0
                            nm = max(k1, k2)
                            for g in range(gpu_num):
                                if g != g1 and g != g2 and kc[g] > nm:
                                    nm = kc[g]
                            if nm < bsv - 1e-12:
                                bsv = nm
                                bs = (g1, ia, i, g2, ib, j)
            if bs is not None:
                g1, ia, i, g2, ib, j = bs
                ri, si = rates[i], sizes[i]
                rj, sj = rates[j], sizes[j]
                gm[g1][ia] = j
                gm[g2][ib] = i
                w[g1] += -ri + rj; s[g1] += -si + sj
                w[g2] += -rj + ri; s[g2] += -sj + si
                kc[g1] = w[g1] / (GPU_MEM_SIZE - s[g1]) if s[g1] < GPU_MEM_SIZE else float('inf')
                kc[g2] = w[g2] / (GPU_MEM_SIZE - s[g2]) if s[g2] < GPU_MEM_SIZE else float('inf')
            else:
                break
        
        final = [0] * n
        for g in range(gpu_num):
            for i in gm[g]:
                final[i] = g
        return final
    
    # MAIN
    start = time.time()
    ba = None
    bk = float('inf')
    
    def ub(a):
        nonlocal ba, bk
        if a is not None:
            mk = compute_max_kvpr(a)
            if mk < bk:
                bk = mk
                ba = a[:]
    
    # Phase 1: Fast heuristics
    idx = list(range(n))
    for sort_fn in [
        lambda i: rates[i],
        lambda i: sizes[i],
        lambda i: rates[i] / max(sizes[i], 0.01),
        lambda i: rates[i] * sizes[i],
    ]:
        for rev in [True, False]:
            order = sorted(idx, key=sort_fn, reverse=rev)
            try:
                ub(greedy_min_kvpr(order))
            except Exception:
                pass
    
    # Phase 2: Binary search with heuristic packing
    tr = sum(rates)
    ts = sum(sizes)
    tl = tr / max(gpu_num * GPU_MEM_SIZE - ts, 0.01)
    th = bk * 1.01 if bk < float('inf') else tl * 100
    
    if ba is None:
        for tt in [tl * 10, tl * 100, tl * 1000]:
            for m in ['best', 'worst']:
                r = try_pack(tt, m)
                if r is not None:
                    ub(r)
            if ba is not None:
                th = bk * 1.01
                break
    
    if ba is None:
        raise ValueError("No valid placement")
    
    for m in ['best', 'worst', 'first']:
        lo, hi = tl * 0.9, th
        for _ in range(60):
            if hi - lo < tl * 1e-10:
                break
            mid = (lo + hi) / 2
            r = try_pack(mid, m)
            if r is not None:
                ub(r)
                hi = mid
            else:
                lo = mid
    
    # Near-optimal packing solutions
    cands = [ba]
    for mult in [0.999, 1.0, 1.001, 1.005, 1.01, 1.05, 1.1]:
        for m in ['best', 'worst', 'first']:
            r = try_pack(bk * mult, m)
            if r is not None:
                cands.append(r)
    
    # Refine heuristic solutions
    seen = set()
    for c in cands:
        k = tuple(c)
        if k not in seen:
            seen.add(k)
            ub(refine(c))
    
    heuristic_best = ba[:]
    heuristic_kvpr = bk
    
    # Phase 3: ILP binary search
    # Scale budget by problem size - small problems can use more time
    if n * gpu_num <= 50:
        total_budget = 15
    elif n * gpu_num <= 150:
        total_budget = 12
    elif n * gpu_num <= 500:
        total_budget = 8
    else:
        total_budget = 4  # Large problems - ILP may not help
    
    elapsed = time.time() - start
    remaining = max(total_budget - elapsed, 1)
    
    if remaining > 2:
        ilp_lo = tl * 0.99
        ilp_hi = bk
        per_solve = max(min(remaining / 15, 3), 1)
        
        for _ in range(20):
            if ilp_hi - ilp_lo < tl * 1e-9:
                break
            if time.time() - start > total_budget:
                break
            mid = (ilp_lo + ilp_hi) / 2
            try:
                # Use best known solution as warm start
                ilp_result = solve_ilp(mid, time_limit=max(int(per_solve), 1), warm_start=ba)
                if ilp_result is not None:
                    ub(ilp_result)
                    ub(refine(ilp_result))
                    ilp_hi = mid
                else:
                    ilp_lo = mid
            except Exception:
                ilp_lo = mid
    
    placement = {g: [] for g in range(gpu_num)}
    for i, g in enumerate(ba):
        placement[g].append(models[i])
    return placement

# EVOLVE-BLOCK-END