import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Hybrid global search for low-makespan transactional schedules.
    Techniques combined:
    - Exact-cost pairwise analysis (singleton costs and deltas).
    - Diverse seeds (RCM variants, precedence, greedy) with local refinement.
    - Guided beam search and bounded A* on prefixes (admissible h=0).
    - Large Neighborhood Search (LNS) driven by high-conflict hotspots:
      destroy a subset around hottest transactions and greedily repair via
      best-position insertions evaluated by the true makespan oracle.
    Returns: (best_makespan, best_sequence)
    """
    from math import exp
    import heapq, time

    n = workload.num_txns
    txns = list(range(n))

    if n == 0:
        return 0, []

    # Memoized exact evaluation for sequences
    eval_cache = {}
    def eval_cost(seq):
        k = tuple(seq)
        v = eval_cache.get(k)
        if v is None:
            v = workload.get_opt_seq_cost(seq)
            eval_cache[k] = v
        return v

    # Singleton and pairwise incremental costs
    c1 = [eval_cost([i]) for i in txns]
    delta = [[0.0] * n for _ in range(n)]
    for i in txns:
        for j in txns:
            if i != j:
                delta[i][j] = eval_cost([i, j]) - c1[i]

    # Pairwise bias: positive means placing j after k is harmful (prefer k before j)
    bias = [[0.0] * n for _ in range(n)]
    for j in txns:
        for k in txns:
            if j != k:
                d = delta[j][k] - delta[k][j]
                bias[j][k] = d if d > 0 else 0.0

    # Precedence order via skew of pairwise deltas
    skew = [0.0] * n
    for i in txns:
        s = 0.0
        for j in txns:
            if i != j:
                s += (delta[j][i] - delta[i][j])
        skew[i] = s
    prec = sorted(txns, key=lambda t: -skew[t])

    # Heuristic for beam expansion: immediate penalty + alpha * future interference on remaining (for ranking only)
    def h_cost(seq, rem, j, alpha):
        inc = 0.0 if not seq else sum(delta[i][j] for i in seq)
        fut = 0.0
        if rem:
            for k in rem:
                if k != j:
                    fut += bias[j][k]
        return inc + alpha * fut

    def beam_run(seed=None, bw=4, top_h=6, rand_k=2, alpha=0.35):
        starts = []
        if seed is not None:
            starts.append(seed)
        if prec:
            starts.append(prec[0])
        seen = set()
        uniq = []
        for s in starts:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        starts = uniq
        rest = [t for t in txns if t not in starts]
        random.shuffle(rest)
        starts.extend(rest[:max(0, bw - len(starts))])

        beam = []
        for s in starts[:max(1, min(bw, n))]:
            seq = [s]
            rem = [t for t in txns if t != s]
            beam.append((seq, rem, eval_cost(seq)))

        while beam and len(beam[0][0]) < n:
            frontier = []
            for seq, rem, cost in beam:
                if not rem:
                    frontier.append((seq, rem, cost))
                    continue
                ranked = sorted(rem, key=lambda j: h_cost(seq, rem, j, alpha))
                cand = ranked[:min(len(ranked), top_h)]
                others = [x for x in rem if x not in cand]
                if others and rand_k > 0:
                    cand += random.sample(others, min(rand_k, len(others)))
                for t in cand:
                    ns = seq + [t]
                    nr = [x for x in rem if x != t]
                    frontier.append((ns, nr, eval_cost(ns)))
            if not frontier:
                break
            frontier.sort(key=lambda x: x[2])
            beam = frontier[:bw]

        if not beam:
            seq = txns[:]
            random.shuffle(seq)
            return eval_cost(seq), seq
        beam.sort(key=lambda x: (len(x[0]) != n, x[2]))
        return beam[0][2], beam[0][0]

    def local_refine(seq, adj_passes=2, insert_trials=50, block_trials=24, long_swaps=14):
        best = seq[:]
        best_c = eval_cost(best)

        # Adjacent swap hill-climb
        for _ in range(adj_passes):
            improved = False
            for i in range(len(best) - 1):
                cand = best[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_c:
                    best, best_c = cand, c
                    improved = True
            if not improved:
                break

        # Block moves
        if len(best) > 3:
            trials = 0
            while trials < block_trials:
                L = 2 if len(best) < 6 else random.choice([2, 3])
                i = random.randrange(0, len(best) - L + 1)
                block = best[i:i + L]
                rem = best[:i] + best[i + L:]
                positions = set()
                base = min(i, len(rem))
                for d in (-3, -2, -1, 0, 1, 2, 3):
                    p = base + d
                    if 0 <= p <= len(rem):
                        positions.add(p)
                while len(positions) < min(8, len(rem) + 1):
                    positions.add(random.randrange(len(rem) + 1))
                improved_here = False
                for p in positions:
                    cand = rem[:p] + block + rem[p:]
                    c = eval_cost(cand)
                    if c < best_c:
                        best, best_c = cand, c
                        improved_here = True
                        break
                trials = 0 if improved_here else trials + 1

        # Targeted single insertions
        tries = 0
        while tries < insert_trials:
            i = random.randrange(len(best))
            x = best[i]
            rem = best[:i] + best[i + 1:]
            pos = set()
            base = min(i, len(rem))
            for d in (-3, -2, -1, 0, 1, 2, 3):
                p = base + d
                if 0 <= p <= len(rem):
                    pos.add(p)
            while len(pos) < min(8, len(rem) + 1):
                pos.add(random.randrange(len(rem) + 1))
            improved_here = False
            for p in pos:
                cand = rem[:p] + [x] + rem[p:]
                c = eval_cost(cand)
                if c < best_c:
                    best, best_c = cand, c
                    improved_here = True
                    break
            tries = 0 if improved_here else tries + 1

        # Sparse long-range swaps with annealed acceptance
        if len(best) > 3:
            T0 = max(1.0, best_c * 0.02)
            for t in range(long_swaps):
                i = random.randrange(len(best))
                j = random.randrange(len(best))
                if i == j or abs(i - j) <= 1:
                    continue
                if i > j:
                    i, j = j, i
                cand = best[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = eval_cost(cand)
                if c < best_c:
                    best, best_c = cand, c
                else:
                    T = T0 * (0.86 ** t)
                    if random.random() < exp(-(c - best_c) / max(1e-9, T)):
                        best, best_c = cand, c

        return best_c, best

    # -------------------------
    # RCM-based candidate seeds
    # -------------------------
    def build_rcm_candidates():
        candidates = []
        # Compute mutual interference weights (symmetric, nonnegative)
        mutual = [[0.0] * n for _ in range(n)]
        pos_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                w = delta[i][j] + delta[j][i]
                w = w if w > 0 else 0.0
                mutual[i][j] = w
                mutual[j][i] = w
                if w > 0:
                    pos_vals.append(w)

        if not pos_vals:
            return candidates

        def build_adj_by_pred(pred):
            adj = [set() for _ in range(n)]
            m = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if pred(i, j):
                        adj[i].add(j)
                        adj[j].add(i)
                        m += 1
            return adj, m

        # Percentiles
        try:
            import numpy as _np
            p50 = float(_np.percentile(pos_vals, 50))
            p65 = float(_np.percentile(pos_vals, 65))
            p80 = float(_np.percentile(pos_vals, 80))
            pct_thresholds = [p50, p65, p80]
        except Exception:
            pos_vals_sorted = sorted(pos_vals)
            pct_thresholds = [pos_vals_sorted[len(pos_vals_sorted) // 2]]
            if len(pos_vals_sorted) > 4:
                pct_thresholds.append(pos_vals_sorted[int(0.65 * (len(pos_vals_sorted) - 1))])
                pct_thresholds.append(pos_vals_sorted[int(0.8 * (len(pos_vals_sorted) - 1))])

        # Top-k per node variants
        ks = []
        if n <= 10:
            ks = [min(n - 1, 3), min(n - 1, 5)]
        else:
            ks = [min(n - 1, max(3, n // 6)), min(n - 1, max(4, n // 4)), min(n - 1, 8)]

        variants = []
        for thr in pct_thresholds:
            def pred(i, j, t=thr):
                return mutual[i][j] >= t and mutual[i][j] > 0
            adj, m = build_adj_by_pred(pred)
            if m > 0:
                variants.append(adj)

        for k in ks:
            adj = [set() for _ in range(n)]
            for i in range(n):
                neigh = [(mutual[i][j], j) for j in range(n) if j != i and mutual[i][j] > 0]
                neigh.sort(reverse=True)
                for _, j in neigh[:k]:
                    adj[i].add(j)
                    adj[j].add(i)
            edges = sum(len(adj[i]) for i in range(n)) // 2
            if edges > 0:
                variants.append(adj)

        def canon(adj):
            es = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        es.append((i, j))
            return tuple(sorted(es))
        seen = set()
        uniq_variants = []
        for adj in variants:
            c = canon(adj)
            if c not in seen:
                seen.add(c)
                uniq_variants.append(adj)

        if not uniq_variants:
            return candidates

        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import reverse_cuthill_mckee as rcm
            have_scipy = True
        except Exception:
            have_scipy = False

        def bfs_rcm(adj):
            deg = [len(adj[i]) for i in range(n)]
            order = []
            visited = [False] * n
            while len(order) < n:
                start = min((i for i in range(n) if not visited[i]), key=lambda x: deg[x])
                queue = [start]
                visited[start] = True
                while queue:
                    u = queue.pop(0)
                    order.append(u)
                    nbrs = [v for v in adj[u] if not visited[v]]
                    nbrs.sort(key=lambda x: deg[x])
                    for v in nbrs:
                        visited[v] = True
                        queue.append(v)
            return order[::-1]

        for adj in uniq_variants:
            if have_scipy:
                rows = []
                cols = []
                data = []
                for i in range(n):
                    for j in adj[i]:
                        rows.append(i)
                        cols.append(j)
                        data.append(1)
                if not rows:
                    continue
                A = csr_matrix((data, (rows, cols)), shape=(n, n))
                try:
                    perm = rcm(A, symmetric_mode=True)
                    seq = list(perm)
                    if len(seq) == n:
                        candidates.append(seq)
                        candidates.append(seq[::-1])
                except Exception:
                    seq = bfs_rcm(adj)
                    candidates.append(seq)
                    candidates.append(seq[::-1])
            else:
                seq = bfs_rcm(adj)
                candidates.append(seq)
                candidates.append(seq[::-1])

        uniq = []
        seen = set()
        for cnd in candidates:
            t = tuple(cnd)
            if t not in seen and len(cnd) == n:
                seen.add(t)
                uniq.append(cnd)
        return uniq

    # Hyperparameters
    bw = 6 if n > 12 else (5 if n > 8 else 4)
    top_h = min(10, max(4, n // 2))
    rand_k = 2 if n > 6 else 1
    restarts = max(3, int(num_seqs))

    # Seed candidates
    candidates = []
    candidates.extend(build_rcm_candidates())
    candidates.append(prec[:])

    # Greedy incremental by true marginal cost
    remaining = set(txns)
    start = min(txns, key=lambda t: c1[t])
    greedy = [start]
    remaining.remove(start)
    while remaining:
        best_t, best_c = None, float("inf")
        for t in list(remaining):
            c = eval_cost(greedy + [t])
            if c < best_c:
                best_c = c
                best_t = t
        greedy.append(best_t)
        remaining.remove(best_t)
    candidates.append(greedy)

    # Local refinement of seeds
    best_cost, best_seq = float("inf"), None
    for cand in candidates:
        _ = eval_cost(cand)
        c1c, s1 = local_refine(cand, adj_passes=2, insert_trials=40, block_trials=18, long_swaps=10)
        if c1c < best_cost:
            best_cost, best_seq = c1c, s1

    # Fallback to precedence if needed
    if best_seq is None:
        base_seq = prec[:]
        best_cost, best_seq = local_refine(base_seq, adj_passes=2, insert_trials=40, block_trials=18, long_swaps=10)

    # Guided multistart beams
    seeds = []
    if prec:
        seeds.append(prec[0])
    if n > 0:
        seeds.append(min(range(n), key=lambda i: c1[i]))
    for r in range(restarts):
        alpha = 0.25 if r % 2 == 0 else 0.4
        seed = seeds[r % len(seeds)] if seeds else None
        c, s = beam_run(seed=seed, bw=bw, top_h=top_h, rand_k=rand_k, alpha=alpha)
        c, s = local_refine(s, adj_passes=2, insert_trials=45, block_trials=20, long_swaps=12)
        if c < best_cost:
            best_cost, best_seq = c, s

    # Diversification: unseeded beam + quick refine
    c3, s3 = beam_run(seed=None, bw=bw, top_h=min(top_h + 1, n), rand_k=rand_k + 1, alpha=0.45)
    c3, s3 = local_refine(s3, adj_passes=1, insert_trials=35, block_trials=16, long_swaps=10)
    if c3 < best_cost:
        best_cost, best_seq = c3, s3

    # Iterated local search
    ils_rounds = 2 if n <= 8 else 3
    cur_cost, cur_seq = best_cost, best_seq[:]
    for _ in range(ils_rounds):
        cand = cur_seq[:]
        if len(cand) > 4 and random.random() < 0.6:
            i = random.randrange(len(cand) - 2)
            j = min(len(cand), i + random.choice([3, 4, 5]))
            blk = cand[i:j]
            blk.reverse()
            cand = cand[:i] + blk + cand[j:]
        else:
            i = random.randrange(len(cand))
            j = random.randrange(len(cand))
            if i != j:
                cand[i], cand[j] = cand[j], cand[i]
        c, s = local_refine(cand, adj_passes=1, insert_trials=30, block_trials=12, long_swaps=8)
        if c < best_cost:
            best_cost, best_seq = c, s
        cur_cost, cur_seq = best_cost, best_seq[:]

    # -----------------------------
    # A* best-first search on prefixes (admissible h=0) with pruning by incumbent
    # -----------------------------
    def astar_best_first(inc_cost, inc_seq):
        # Budget tuned to be robust across workloads
        max_nodes = 1500 + 200 * n
        time_budget = min(2.5 + 0.06 * n, 8.0)
        start_time = time.time()
        expanded = 0

        # Tie-breaker: sum of harmful biases from placed to remaining (lower is better)
        def tie_score(seq, rem):
            if not seq or not rem:
                return 0.0
            s = 0.0
            for i in seq:
                bi = bias[i]
                for j in rem:
                    s += bi[j]
            return s

        # Initial frontier: choose diverse promising starts
        starts = []
        # Always include best singleton by c1 and precedence first
        starts.append(min(range(n), key=lambda i: c1[i]) if n > 0 else 0)
        if prec:
            starts.append(prec[0])
        # Add a few additional good starts by skew and random
        add_more = sorted(range(n), key=lambda i: -skew[i])[:min(4, n)]
        for s in add_more:
            starts.append(s)
        if n > 6:
            starts.extend(random.sample([t for t in range(n)], min(2, n)))
        # Dedup
        seen_s = []
        for s in starts:
            if s not in seen_s:
                seen_s.append(s)
        starts = seen_s[:min(len(seen_s), max(5, min(8, n)))]

        pq = []
        counter = 0
        best_g_prefix = {}  # key: tuple(prefix) -> best g

        for s in starts:
            seq = [s]
            rem = [t for t in txns if t != s]
            g = c1[s]
            if g >= inc_cost:
                continue
            key = tuple(seq)
            best_g_prefix[key] = g
            f = g  # h = 0
            heappush = heapq.heappush
            heappop = heapq.heappop
            heappush(pq, (f, tie_score(seq, rem), -len(seq), counter, seq, rem, g))
            counter += 1

        best_local_cost = inc_cost
        best_local_seq = inc_seq[:]

        while pq and expanded < max_nodes and (time.time() - start_time) < time_budget:
            f, tie, negdepth, _, seq, rem, g = heapq.heappop(pq)
            if g >= best_local_cost:
                continue
            if not rem:
                # Complete schedule
                if g < best_local_cost:
                    best_local_cost, best_local_seq = g, seq[:]
                continue

            expanded += 1

            # Expand children; rank candidates by lookahead to focus promising branches
            ranked = sorted(rem, key=lambda j: h_cost(seq, rem, j, 0.35))
            top_k = min(len(ranked), 6 if n > 10 else 8)
            cand_next = ranked[:top_k]
            others = [x for x in rem if x not in cand_next]
            if others and len(cand_next) < len(rem) and top_k < len(rem) and random.random() < 0.3:
                cand_next += random.sample(others, min(2, len(others)))

            for t in cand_next:
                ns = seq + [t]
                nr = [x for x in rem if x != t]
                g2 = eval_cost(ns)
                if g2 >= best_local_cost:
                    continue
                key = tuple(ns)
                if g2 >= best_g_prefix.get(key, float("inf")):
                    continue
                best_g_prefix[key] = g2
                f2 = g2  # h=0
                heapq.heappush(pq, (f2, tie_score(ns, nr), -len(ns), counter, ns, nr, g2))
                counter += 1

        return best_local_cost, best_local_seq

    astar_cost, astar_seq = astar_best_first(best_cost, best_seq)
    if astar_cost < best_cost:
        best_cost, best_seq = astar_cost, astar_seq

    # -------------------------------------------------
    # Large Neighborhood Search (LNS) driven by hotspots
    # -------------------------------------------------
    # Build symmetric mutual interference weights and per-txn heat
    mutual = [[0.0] * n for _ in range(n)]
    heat = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            w = delta[i][j] + delta[j][i]
            if w > 0:
                mutual[i][j] = w
                s += w
        heat[i] = s

    def pick_destroy_set(cur_seq, target_size):
        # Prefer high-heat anchors and their strongest neighbors
        m = max(1, min(target_size, n - 1))
        # Choose anchors among top-q hot txns, with randomness
        order_hot = sorted(range(n), key=lambda t: heat[t], reverse=True)
        q = min(n, max(6, m * 2))
        pool = order_hot[:q]
        anchors_cnt = 1 if n < 8 else (2 if n < 16 else 3)
        anchors_cnt = min(anchors_cnt, m)
        anchors = set(random.sample(pool, anchors_cnt))
        removed = set(anchors)
        # Add strongest neighbors iteratively
        by_pos = {t: idx for idx, t in enumerate(cur_seq)}
        cand_neighbors = []
        for a in anchors:
            neigh = [(mutual[a][j], j) for j in range(n) if j != a and mutual[a][j] > 0]
            neigh.sort(reverse=True)
            # bias neighbors that are close in the current order to capture local clusters
            for w, j in neigh[:max(3, m)]:
                bias_closeness = 1.0 / (1 + abs(by_pos.get(a, 0) - by_pos.get(j, 0)))
                cand_neighbors.append((w * (1.0 + 0.25 * bias_closeness), j))
        cand_neighbors.sort(reverse=True)
        for _, j in cand_neighbors:
            if len(removed) >= m:
                break
            removed.add(j)
        # If still short, fill with additional hot txns
        for t in order_hot:
            if len(removed) >= m:
                break
            removed.add(t)
        # Last resort: random fill
        if len(removed) < m:
            rest = [t for t in range(n) if t not in removed]
            random.shuffle(rest)
            for t in rest:
                removed.add(t)
                if len(removed) >= m:
                    break
        # Never remove all
        if len(removed) >= n:
            removed = set(list(removed)[:n - 1])
        return removed

    def greedy_repair(base_seq, removed_set):
        seq = base_seq[:]
        # Insert higher-heat txns first to place hot ones optimally
        to_insert = sorted(list(removed_set), key=lambda t: heat[t], reverse=True)
        for t in to_insert:
            best_pos = 0
            best_c = float("inf")
            # Evaluate all insertion positions
            for pos in range(len(seq) + 1):
                cand = seq[:pos] + [t] + seq[pos:]
                c = eval_cost(cand)
                if c < best_c:
                    best_c = c
                    best_pos = pos
            seq = seq[:best_pos] + [t] + seq[best_pos:]
        return eval_cost(seq), seq

    def lns_hot_conflict(seed_seq, seed_cost):
        cur_seq = seed_seq[:]
        cur_cost = seed_cost
        best_seq_lns = seed_seq[:]
        best_cost_lns = seed_cost
        # Rounds and time budget
        rounds = min(18, 8 + n // 3)
        time_budget = min(2.5 + 0.05 * n, 7.0)
        start_time = time.time()
        # Annealing temperature
        T0 = max(1.0, best_cost_lns * 0.01)
        alpha = 0.9
        for it in range(rounds):
            if (time.time() - start_time) > time_budget:
                break
            # Adaptive destroy size: 10–30% with jitter
            frac = 0.12 + 0.18 * random.random()
            m = max(1, min(n - 1, int(round(n * frac))))
            removed = pick_destroy_set(cur_seq, m)
            base = [t for t in cur_seq if t not in removed]
            cand_cost, cand_seq = greedy_repair(base, removed)
            # Accept if better; else probabilistic acceptance
            if cand_cost < cur_cost:
                cur_seq, cur_cost = cand_seq, cand_cost
                if cand_cost < best_cost_lns:
                    best_cost_lns, best_seq_lns = cand_cost, cand_seq
            else:
                T = T0 * (alpha ** it)
                if random.random() < exp(-(cand_cost - cur_cost) / max(1e-9, T)):
                    cur_seq, cur_cost = cand_seq, cand_cost
        return best_cost_lns, best_seq_lns

    # Run LNS from the best sequence found so far
    lns_cost, lns_seq = lns_hot_conflict(best_seq, best_cost)
    if lns_cost < best_cost:
        best_cost, best_seq = lns_cost, lns_seq

    # Hierarchical community-based scheduling (community detection + intra-cluster optimization + inter-cluster ordering)
    # This implements the breakthrough idea:
    # - Build weighted conflict graph (mutual interference)
    # - Detect communities (greedy modularity)
    # - Optimize each small community exactly/near-exactly; large ones via subset beam + local refine
    # - Order communities with directed-harm precedence and small brute force/greedy
    # - Smooth cross-cluster boundaries and final light global refine
    def hierarchical_schedule():
        try:
            import networkx as nx
            from networkx.algorithms.community import greedy_modularity_communities
        except Exception:
            return None

        # Build symmetric nonnegative conflict graph
        G = nx.Graph()
        G.add_nodes_from(txns)
        edge_cnt = 0
        for i in range(n):
            di = delta[i]
            for j in range(i + 1, n):
                w = di[j] + delta[j][i]
                if w > 0:
                    G.add_edge(i, j, weight=w)
                    edge_cnt += 1
        if edge_cnt == 0:
            return None

        # Detect communities
        communities = list(greedy_modularity_communities(G, weight="weight"))
        clusters = [list(sorted(c)) for c in communities]

        # Merge tiny communities to avoid fragmentation
        min_sz = 2 if n >= 4 else 1

        def inter_w(ca, cb):
            s = 0.0
            for a in ca:
                ra = delta[a]
                for b in cb:
                    w = ra[b] + delta[b][a]
                    if w > 0:
                        s += w
            return s

        while len(clusters) > 1 and any(len(c) < min_sz for c in clusters):
            idx_small = min(range(len(clusters)), key=lambda t: len(clusters[t]))
            c = clusters.pop(idx_small)
            # Merge small cluster into the most connected neighbor by total weight
            jbest, wbest = None, -1.0
            for j in range(len(clusters)):
                w = inter_w(c, clusters[j])
                if w > wbest:
                    wbest, jbest = w, j
            clusters[jbest] = clusters[jbest] + c

        # If trivial clustering, skip
        if len(clusters) <= 1:
            return None

        # Optimize within a community
        def optimize_cluster(sub):
            m = len(sub)
            if m <= 9:
                # Best-first search (admissible h=0) with pruning by incumbent and small time cap
                import heapq, time as _t, random as _rnd
                start_t = _t.time()
                time_cap = 1.0 + 0.05 * m
                best_c, best_s = float("inf"), None
                # Start nodes: top by singleton cost and skew
                seeds = sorted(sub, key=lambda t: c1[t])[:min(2, m)]
                top_sk = sorted(sub, key=lambda t: -skew[t])[:min(2, m)]
                for t in top_sk:
                    if t not in seeds:
                        seeds.append(t)
                pq = []
                counter = 0
                for s in seeds:
                    seq = [s]
                    rem = [t for t in sub if t != s]
                    g = eval_cost(seq)
                    heapq.heappush(pq, (g, -len(seq), counter, seq, rem))
                    counter += 1
                visited = {}
                while pq:
                    g, _, _, seq, rem = heapq.heappop(pq)
                    if g >= best_c:
                        continue
                    if not rem:
                        best_c, best_s = g, seq[:]
                        continue
                    ranked = sorted(rem, key=lambda j: sum(delta[i][j] for i in seq))
                    cand = ranked[:min(len(ranked), 4)]
                    others = [x for x in rem if x not in cand]
                    if others:
                        cand += _rnd.sample(others, min(2, len(others)))
                    for t in cand:
                        ns = seq + [t]
                        key = tuple(ns)
                        g2 = eval_cost(ns)
                        if g2 >= best_c:
                            continue
                        if g2 >= visited.get(key, float("inf")):
                            continue
                        visited[key] = g2
                        nr = [x for x in rem if x != t]
                        heapq.heappush(pq, (g2, -len(ns), counter, ns, nr))
                        counter += 1
                    if (_t.time() - start_t) > time_cap and m >= 9:
                        break
                if best_s is None:
                    # Fallback greedy within cluster
                    seq = []
                    r = sub[:]
                    while r:
                        bt, bc = None, float("inf")
                        for t in r:
                            c = eval_cost(seq + [t])
                            if c < bc:
                                bc, bt = c, t
                        seq.append(bt)
                        r.remove(bt)
                    best_s = seq
                # Light local refine within cluster
                _, best_s = local_refine(best_s, adj_passes=1, insert_trials=12, block_trials=6, long_swaps=4)
                return best_s
            else:
                # Larger cluster: run a small beam-search restricted to the subset
                import random as _rnd
                bw_loc = 4 if m < 16 else 5
                top_h_loc = min(8, max(4, m // 2))
                seeds = []
                s1 = min(sub, key=lambda t: c1[t])
                seeds.append(s1)
                sk = sorted(sub, key=lambda t: -skew[t])[0]
                if sk not in seeds:
                    seeds.append(sk)
                beam = []
                for s in seeds:
                    seq = [s]
                    rem = [t for t in sub if t != s]
                    beam.append((seq, rem, eval_cost(seq)))
                while beam and len(beam[0][0]) < m:
                    frontier = []
                    for seq, rem, _c in beam:
                        if not rem:
                            frontier.append((seq, rem, _c))
                            continue
                        ranked = sorted(rem, key=lambda j: h_cost(seq, rem, j, 0.35))
                        cand = ranked[:min(len(ranked), top_h_loc)]
                        others = [x for x in rem if x not in cand]
                        if others:
                            cand += _rnd.sample(others, min(1, len(others)))
                        for t in cand:
                            ns = seq + [t]
                            nr = [x for x in rem if x != t]
                            frontier.append((ns, nr, eval_cost(ns)))
                    if not frontier:
                        break
                    frontier.sort(key=lambda x: x[2])
                    beam = frontier[:bw_loc]
                beam.sort(key=lambda x: (len(x[0]) != m, x[2]))
                seq = beam[0][0]
                _, seq = local_refine(seq, adj_passes=1, insert_trials=20, block_trials=10, long_swaps=6)
                return seq

        cluster_orders = [optimize_cluster(c[:]) for c in clusters]
        C = len(cluster_orders)

        # Directed harm between clusters: D[a][b] = harm if a before b
        D = [[0.0] * C for _ in range(C)]
        for a in range(C):
            for b in range(C):
                if a == b:
                    continue
                s_h = 0.0
                for i in cluster_orders[a]:
                    for j in cluster_orders[b]:
                        d = delta[i][j] - delta[j][i]
                        if d > 0:
                            s_h += d
                D[a][b] = s_h

        # Precedence score per cluster
        scores = [sum(D[i][k] for k in range(C)) - sum(D[k][i] for k in range(C)) for i in range(C)]
        base_order = sorted(range(C), key=lambda i: -scores[i])

        # Choose cluster order: brute force if small, else greedy incremental by true cost
        import itertools
        best_global_seq, best_global_cost = None, float("inf")
        if C <= 7:
            for ord_idx in itertools.permutations(range(C)):
                seq = []
                for idx in ord_idx:
                    seq += cluster_orders[idx]
                c = eval_cost(seq)
                if c < best_global_cost:
                    best_global_cost, best_global_seq = c, seq
        else:
            # Greedy by minimal true makespan prefix
            used = set()
            order = []
            # Start with best singleton cluster by sum of singleton costs
            start_idx = min(range(C), key=lambda ci: sum(c1[t] for t in cluster_orders[ci]))
            order.append(start_idx)
            used.add(start_idx)
            while len(order) < C:
                best_next, bc = None, float("inf")
                for ci in range(C):
                    if ci in used:
                        continue
                    cand = []
                    for idx in order + [ci]:
                        cand += cluster_orders[idx]
                    cc = eval_cost(cand)
                    if cc < bc:
                        bc, best_next = cc, ci
                order.append(best_next)
                used.add(best_next)
            seq = []
            for idx in order:
                seq += cluster_orders[idx]
            best_global_seq = seq
            best_global_cost = eval_cost(seq)

        # Cross-boundary smoothing: limited windowed insertions/swaps across cluster boundaries
        def boundary_refine(seq, segments, window=2, rounds=2):
            # segments: lengths of clusters in the chosen order
            pos = 0
            bounds = []
            for L in segments:
                bounds.append((pos, pos + L))
                pos += L
            best_seq = seq[:]
            best_c = eval_cost(best_seq)
            for _ in range(rounds):
                improved = False
                for bi in range(len(bounds) - 1):
                    l0, l1 = bounds[bi]
                    r0, r1 = bounds[bi + 1]
                    left_tail = list(range(max(l1 - window, l0), l1))
                    right_head = list(range(r0, min(r0 + window, r1)))
                    # Swap across boundary
                    if left_tail and right_head:
                        i = left_tail[-1]
                        j = right_head[0]
                        cand = best_seq[:]
                        cand[i], cand[j] = cand[j], cand[i]
                        c = eval_cost(cand)
                        if c < best_c:
                            best_seq, best_c = cand, c
                            improved = True
                            continue
                    # Move right -> left end
                    for j in right_head:
                        cand = best_seq[:]
                        x = cand.pop(j)
                        cand.insert(l1, x)
                        c = eval_cost(cand)
                        if c < best_c:
                            best_seq, best_c = cand, c
                            improved = True
                            break
                    if improved:
                        continue
                    # Move left tail -> right beginning
                    for i in reversed(left_tail):
                        cand = best_seq[:]
                        x = cand.pop(i)
                        cand.insert(r0, x)
                        c = eval_cost(cand)
                        if c < best_c:
                            best_seq, best_c = cand, c
                            improved = True
                            break
                if not improved:
                    break
            return best_c, best_seq

        # Derive segments for the chosen concatenation by scanning cluster IDs
        idx_of_cluster = {}
        for ci, cl in enumerate(cluster_orders):
            for t in cl:
                idx_of_cluster[t] = ci
        segments = []
        cur_ci, cur_len = None, 0
        for t in best_global_seq:
            ci = idx_of_cluster[t]
            if cur_ci is None:
                cur_ci, cur_len = ci, 1
            elif ci == cur_ci:
                cur_len += 1
            else:
                segments.append(cur_len)
                cur_ci, cur_len = ci, 1
        if cur_ci is not None:
            segments.append(cur_len)

        bc, bseq = boundary_refine(best_global_seq, segments, window=2, rounds=2)
        # Final light global refinement
        bc2, bseq2 = local_refine(bseq, adj_passes=1, insert_trials=15, block_trials=10, long_swaps=6)
        return bc2, bseq2

    hs = hierarchical_schedule()
    if hs is not None:
        hcost, hseq = hs
        if hcost < best_cost:
            best_cost, best_seq = hcost, hseq

    return best_cost, best_seq

# EVOLVE-BLOCK-END

def get_random_costs():
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
    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3]


if __name__ == "__main__":
    makespan, schedule = get_random_costs()
    print(f"Makespan: {makespan}")
