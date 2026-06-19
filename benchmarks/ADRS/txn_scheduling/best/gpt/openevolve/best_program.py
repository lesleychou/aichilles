import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan schedule via multi-start constructive heuristics with insertion-based local search.
    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # Lightweight memoization to reduce repeated cost evaluations
    cost_cache = {}

    def cost_of(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(seq)
            cost_cache[key] = c
        return c

    def build_seq_greedy_ends(num_samples, sample_rate):
        # Greedy double-ended construction: place each next txn either at front or back
        seq = []
        remaining = list(range(workload.num_txns))
        start = random.choice(remaining)
        seq.append(start)
        remaining.remove(start)
        if remaining:
            k = min(num_samples, len(remaining))
            candidates = random.sample(remaining, k=k)
            best_t = None
            best_cost = None
            place_front = False
            for t in candidates:
                c_end = cost_of(seq + [t])
                c_front = cost_of([t] + seq)
                m = min(c_end, c_front)
                if best_cost is None or m < best_cost:
                    best_cost = m
                    best_t = t
                    place_front = (c_front <= c_end)
            if best_t is None:
                best_t = random.choice(remaining)
                place_front = random.random() < 0.5
            if place_front:
                seq = [best_t] + seq
            else:
                seq.append(best_t)
            remaining.remove(best_t)
        while remaining:
            if random.random() > sample_rate:
                # Exploration step: random placement at a random end
                t = random.choice(remaining)
                if random.random() < 0.5:
                    seq.insert(0, t)
                else:
                    seq.append(t)
                remaining.remove(t)
                continue
            k = min(num_samples, len(remaining))
            candidates = random.sample(remaining, k=k)
            best_t = None
            best_cost = None
            place_front = False
            for t in candidates:
                c_end = cost_of(seq + [t])
                c_front = cost_of([t] + seq)
                m = min(c_end, c_front)
                if best_cost is None or m < best_cost:
                    best_cost = m
                    best_t = t
                    place_front = (c_front <= c_end)
            if best_t is None:
                best_t = random.choice(remaining)
                place_front = random.random() < 0.5
            if place_front:
                seq = [best_t] + seq
            else:
                seq.append(best_t)
            remaining.remove(best_t)
        return cost_of(seq), seq

    def build_seq_best_insertion(sample_k=6):
        # Best-position insertion: for a sampled next txn, try all positions and pick best
        remaining = list(range(workload.num_txns))
        a = random.choice(remaining)
        remaining.remove(a)
        b = random.choice(remaining)
        remaining.remove(b)
        if cost_of([a, b]) <= cost_of([b, a]):
            seq = [a, b]
        else:
            seq = [b, a]
        while remaining:
            k = min(sample_k, len(remaining))
            candidates = random.sample(remaining, k=k)
            best_seq = None
            best_cost = None
            best_t = None
            for t in candidates:
                for pos in range(len(seq) + 1):
                    new_seq = seq[:pos] + [t] + seq[pos:]
                    c = cost_of(new_seq)
                    if best_cost is None or c < best_cost:
                        best_cost = c
                        best_seq = new_seq
                        best_t = t
            if best_seq is None:
                t = random.choice(remaining)
                pos = random.randrange(0, len(seq) + 1)
                best_seq = seq[:pos] + [t] + seq[pos:]
                best_cost = cost_of(best_seq)
                best_t = t
            seq = best_seq
            remaining.remove(best_t)
        return cost_of(seq), seq

    def local_improve(seq, current_cost, max_passes=2, random_swaps=15):
        # Insertion-based first-improvement passes
        passes = 0
        improved = True
        while improved and passes < max_passes:
            improved = False
            for i in range(len(seq)):
                x = seq[i]
                base = seq[:i] + seq[i + 1:]
                for j in range(len(base) + 1):
                    cand = base[:j] + [x] + base[j:]
                    c = cost_of(cand)
                    if c < current_cost:
                        seq = cand
                        current_cost = c
                        improved = True
                        break
                if improved:
                    break
            passes += 1
        # Adjacent swap pass
        passes2 = 0
        while passes2 < max_passes:
            swapped = False
            for i in range(len(seq) - 1):
                new_seq = seq.copy()
                new_seq[i], new_seq[i + 1] = new_seq[i + 1], new_seq[i]
                c = cost_of(new_seq)
                if c < current_cost:
                    seq = new_seq
                    current_cost = c
                    swapped = True
                    break
            if not swapped:
                break
            passes2 += 1
        # Random pairwise swap exploration
        tries = 0
        while tries < random_swaps:
            i = random.randrange(0, len(seq))
            j = random.randrange(0, len(seq))
            if i == j:
                tries += 1
                continue
            if i > j:
                i, j = j, i
            new_seq = seq.copy()
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            c = cost_of(new_seq)
            if c < current_cost:
                seq = new_seq
                current_cost = c
                tries = 0
            else:
                tries += 1
        return current_cost, seq

    best_cost = float('inf')
    best_seq = None

    # Multiple randomized restarts exploring different constructive heuristics
    tries = max(4, int(num_seqs) if isinstance(num_seqs, int) else 5)
    for _ in range(tries):
        if random.random() < 0.5:
            sample_rate = 0.7 + 0.25 * random.random()
            num_samples = max(4, min(10, 5 + int(5 * random.random())))
            cost, seq = build_seq_greedy_ends(num_samples, sample_rate)
        else:
            cost, seq = build_seq_best_insertion(sample_k=6)
        cost, seq = local_improve(seq, cost, max_passes=2, random_swaps=15)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

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
