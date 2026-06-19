import time
import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


def get_best_schedule(workload, num_seqs):
    """
    V14: Optimized Hybrid SA-LAHC.
    Best config from tuning:
    - T_start=0.018, T_end=0.008
    - LAHC_len=n*3 (dual acceptance criterion)
    - Stagnation restart with reheat
    - 22% construction, 13% LS, 55% SA-LAHC, 5% best-position, 5% ILS
    - Move mix: 8% adj-swap, 12% swap, 58% insert, 12% 2opt, 10% or-opt
    """
    import time as _time
    import math
    start_time = _time.time()
    n = workload.num_txns
    time_budget = 19.0
    
    eval_cost = workload.get_opt_seq_cost

    # ===== Build conflict info =====
    conflict_order = None
    try:
        wl_data = None
        for attr in ['workload', 'txns', 'transactions', '_workload', 'data', '_txns']:
            if hasattr(workload, attr):
                candidate = getattr(workload, attr)
                if isinstance(candidate, dict):
                    wl_data = candidate
                    break
        
        if wl_data is not None:
            txn_write_set = {}
            txn_read_set = {}
            for txn_name, ops_str in wl_data.items():
                txn_id = int(txn_name.replace('txn', ''))
                ops = ops_str.split()
                txn_write_set[txn_id] = set()
                txn_read_set[txn_id] = set()
                for op in ops:
                    parts = op.split('-')
                    if parts[0] == 'w':
                        txn_write_set[txn_id].add(int(parts[1]))
                    else:
                        txn_read_set[txn_id].add(int(parts[1]))
            
            txn_conflict = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    wi = txn_write_set.get(i, set())
                    wj = txn_write_set.get(j, set())
                    ri = txn_read_set.get(i, set())
                    rj = txn_read_set.get(j, set())
                    s = len(wi & wj) * 2 + len(ri & wj) + len(wi & rj)
                    txn_conflict[i] += s
                    txn_conflict[j] += s
            
            conflict_order = sorted(range(n), key=lambda x: -txn_conflict[x])
            print(f"  n={n}, max_conflict={max(txn_conflict)}, min={min(txn_conflict)}")
    except Exception as e:
        print(f"  No conflict info: {e}")
    
    # ===== Construction =====
    def cheapest_insertion(start_txn, max_positions=8, use_conflict_order=False):
        seq = [start_txn]
        remaining = list(range(n))
        remaining.remove(start_txn)
        
        if use_conflict_order and conflict_order is not None:
            remaining = [t for t in conflict_order if t != start_txn]
        else:
            random.shuffle(remaining)
        
        for txn in remaining:
            best_c = float('inf')
            best_p = 0
            npos = len(seq) + 1
            
            if npos <= max_positions:
                positions = range(npos)
            else:
                pset = {0, npos - 1, npos // 2}
                while len(pset) < max_positions:
                    pset.add(random.randint(0, npos - 1))
                positions = sorted(pset)
            
            for p in positions:
                c = eval_cost(seq[:p] + [txn] + seq[p:])
                if c < best_c:
                    best_c = c
                    best_p = p
            seq.insert(best_p, txn)
        
        return eval_cost(seq), seq
    
    def two_sided_greedy(start_txn, num_cand):
        seq = [start_txn]
        remaining = list(range(n))
        remaining.remove(start_txn)
        
        while remaining:
            if len(remaining) <= num_cand:
                cands = remaining[:]
            else:
                cands = random.sample(remaining, num_cand)
            
            best_c = float('inf')
            best_t = cands[0]
            best_front = False
            
            for t in cands:
                cb = eval_cost(seq + [t])
                if cb < best_c:
                    best_c = cb; best_t = t; best_front = False
                cf = eval_cost([t] + seq)
                if cf < best_c:
                    best_c = cf; best_t = t; best_front = True
            
            if best_front:
                seq.insert(0, best_t)
            else:
                seq.append(best_t)
            remaining.remove(best_t)
        
        return eval_cost(seq), seq
    
    # ===== Local Search (Insert) - First improvement =====
    def local_search_insert(seq, cost, time_limit):
        bc = cost
        bs = seq[:]
        improved = True
        while improved and _time.time() < time_limit:
            improved = False
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                if _time.time() >= time_limit:
                    break
                txn = bs[i]
                rem = bs[:i] + bs[i+1:]
                positions = list(range(n))
                random.shuffle(positions)
                for j in positions:
                    if j == i:
                        continue
                    if _time.time() >= time_limit:
                        break
                    ns = rem[:j] + [txn] + rem[j:]
                    nc = eval_cost(ns)
                    if nc < bc:
                        bc = nc; bs = ns; improved = True; break
                if improved:
                    break
        return bc, bs

    # ===== Best-position insert LS =====
    def best_position_insert(seq, cost, time_limit):
        bc = cost
        bs = seq[:]
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            if _time.time() >= time_limit:
                break
            txn = bs[i]
            rem = bs[:i] + bs[i+1:]
            best_j = i
            best_nc = bc
            if n <= 20:
                positions = list(range(n))
            else:
                pset = {0, n-1, n//2, max(0,i-1), min(n-1,i+1)}
                while len(pset) < min(n, 12):
                    pset.add(random.randint(0, n-1))
                positions = list(pset)
            
            for j in positions:
                if _time.time() >= time_limit:
                    break
                ns = rem[:j] + [txn] + rem[j:]
                nc = eval_cost(ns)
                if nc < best_nc:
                    best_nc = nc
                    best_j = j
            
            if best_nc < bc:
                bc = best_nc
                bs = rem[:best_j] + [txn] + rem[best_j:]
        return bc, bs
    
    # ===== Hybrid SA-LAHC =====
    def hybrid_sa_lahc(seq, cost, time_limit):
        lahc_len = max(40, n * 3)
        
        cc = cost
        cs = seq[:]
        bc = cost
        bs = seq[:]
        
        T_start = cost * 0.018
        T_end = 0.008
        
        total_time = time_limit - _time.time()
        if total_time <= 0:
            return bc, bs
        
        log_ratio = math.log(T_end / max(T_start, 0.001))
        sa_start = _time.time()
        iters = 0
        acc = 0
        impr = 0
        
        win = 80
        recent_acc = 0
        T_mult = 1.0
        
        last_impr_iter = 0
        stagnation_limit = max(200, n * 10)
        
        history = [cost] * lahc_len
        
        while True:
            now = _time.time()
            if now >= time_limit:
                break
            
            frac = min((now - sa_start) / total_time, 0.999)
            T = T_start * math.exp(log_ratio * frac) * T_mult
            
            if iters - last_impr_iter > stagnation_limit:
                cc = bc
                cs = bs[:]
                T_mult = min(T_mult * 1.5, 3.0)
                last_impr_iter = iters
                history = [bc] * lahc_len
            
            r = random.random()
            
            if r < 0.08:
                i = random.randint(0, n - 2)
                ns = cs[:]
                ns[i], ns[i+1] = ns[i+1], ns[i]
            elif r < 0.20:
                i, j = random.sample(range(n), 2)
                ns = cs[:]
                ns[i], ns[j] = ns[j], ns[i]
            elif r < 0.78:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 2)
                if j >= i:
                    j += 1
                txn = cs[i]
                ns = cs[:i] + cs[i+1:]
                j = max(0, min(j, len(ns)))
                ns = ns[:j] + [txn] + ns[j:]
            elif r < 0.90:
                i = random.randint(0, n - 2)
                seg_len = random.randint(2, min(n - i, max(2, n // 5)))
                j = i + seg_len
                ns = cs[:i] + cs[i:j][::-1] + cs[j:]
            else:
                sl = random.choice([2, 3])
                if n > sl + 1:
                    i = random.randint(0, n - sl)
                    seg = cs[i:i+sl]
                    rem = cs[:i] + cs[i+sl:]
                    j = random.randint(0, len(rem))
                    ns = rem[:j] + seg + rem[j:]
                else:
                    i = random.randint(0, n - 2)
                    ns = cs[:]
                    ns[i], ns[i+1] = ns[i+1], ns[i]
            
            nc = eval_cost(ns)
            delta = nc - cc
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 0.0001)):
                cc = nc
                cs = ns
                acc += 1
                recent_acc += 1
                if cc < bc:
                    bc = cc
                    bs = cs[:]
                    impr += 1
                    last_impr_iter = iters
            
            iters += 1
            if iters % win == 0:
                ar = recent_acc / win
                if ar < 0.12:
                    T_mult = min(T_mult * 1.15, 3.0)
                elif ar > 0.40:
                    T_mult = max(T_mult * 0.85, 0.3)
                recent_acc = 0
        
        print(f"  SA-LAHC: {iters}it, {acc}acc, {impr}impr, T_mult={T_mult:.2f}, best={bc}")
        return bc, bs
    
    def perturb(seq, strength=3):
        s = seq[:]
        for _ in range(strength):
            i, j = random.sample(range(n), 2)
            s[i], s[j] = s[j], s[i]
        return s
    
    # ===== MAIN =====
    
    # Phase 1: Greedy construction (22%)
    population = []
    p1_end = start_time + time_budget * 0.22
    nr = 0
    while _time.time() < p1_end:
        st = random.randint(0, n - 1)
        if nr == 0:
            c, s = cheapest_insertion(st, max_positions=min(n + 1, 10), use_conflict_order=True)
        elif nr == 1:
            c, s = two_sided_greedy(st, n)
        elif nr == 2:
            c, s = cheapest_insertion(st, max_positions=min(n + 1, 8))
        elif nr < 5:
            c, s = cheapest_insertion(st, max_positions=min(n + 1, 6), use_conflict_order=(nr % 2 == 0))
        else:
            c, s = two_sided_greedy(st, random.choice([10, 15, min(n, 20)]))
        nr += 1
        population.append((c, s))
        population.sort(key=lambda x: x[0])
        if len(population) > 5:
            population = population[:5]
    
    print(f"Phase 1: {nr} restarts, top = {[round(p[0], 2) for p in population[:3]]}")
    
    # Phase 2: LS on top 2 solutions (13%)
    p2_mid = start_time + time_budget * 0.30
    bc = population[0][0]
    bs = population[0][1][:]
    bc, bs = local_search_insert(bs, bc, p2_mid)
    
    if len(population) > 1:
        p2_end = start_time + time_budget * 0.35
        c2 = population[1][0]
        s2 = population[1][1][:]
        c2, s2 = local_search_insert(s2, c2, p2_end)
        if c2 < bc:
            bc = c2; bs = s2
    print(f"Phase 2 (LS): cost = {bc}")
    
    # Phase 3: SA (50%)
    p3_end = start_time + time_budget * 0.85
    bc, bs = hybrid_sa_lahc(bs, bc, p3_end)
    print(f"Phase 3 (SA): cost = {bc}")
    
    # Phase 4: Final LS (10%)
    p4a_end = start_time + time_budget * 0.92
    bc, bs = local_search_insert(bs, bc, p4a_end)
    
    # Quick swap LS
    p4b_end = start_time + time_budget * 0.95
    imp = True
    while imp and _time.time() < p4b_end:
        imp = False
        indices = list(range(n))
        random.shuffle(indices)
        for ii in range(n):
            if _time.time() >= p4b_end:
                break
            i = indices[ii]
            for jj in range(ii + 1, n):
                j = indices[jj]
                if _time.time() >= p4b_end:
                    break
                ns = bs[:]
                ns[i], ns[j] = ns[j], ns[i]
                nc = eval_cost(ns)
                if nc < bc:
                    bc = nc; bs = ns; imp = True; break
            if imp:
                break
    print(f"Phase 4 (final LS): cost = {bc}")
    
    # Phase 5: ILS (5%)
    p5_end = start_time + time_budget
    while _time.time() < p5_end:
        rt = p5_end - _time.time()
        if rt < 0.3:
            break
        p = perturb(bs, strength=random.randint(2, 4))
        pc = eval_cost(p)
        qe = _time.time() + min(rt * 0.7, 1.0)
        pc, p = local_search_insert(p, pc, qe)
        if pc < bc:
            bc = pc; bs = p[:]; print(f"  ILS: {bc}")
    
    elapsed = _time.time() - start_time
    print(f"Total: {elapsed:.2f}s, final cost = {bc}")
    return bc, bs


def get_random_costs():
    start_time = time.time()
    workload_size = 100
    workload = Workload(WORKLOAD_1)

    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)
    print(cost1, cost2, cost3)
    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3], time.time() - start_time