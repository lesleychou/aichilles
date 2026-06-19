import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using beam search with conflict-aware heuristics.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    def compute_conflict_score(txn_seq, candidate):
        """Compute actual makespan if we add candidate to sequence."""
        test_seq = txn_seq + [candidate]
        return workload.get_opt_seq_cost(test_seq)
    
    def beam_search_schedule(beam_width=3, sample_factor=2.0):
        """Build schedule using beam search to maintain multiple promising paths."""
        # Initialize beam with diverse starting transactions
        beam = []
        start_candidates = list(range(workload.num_txns))
        random.shuffle(start_candidates)
        
        # Start with more candidates for better diversity
        num_starts = min(beam_width * 3, workload.num_txns)
        for start_txn in start_candidates[:num_starts]:
            remaining = [x for x in range(workload.num_txns) if x != start_txn]
            beam.append(([start_txn], remaining, workload.get_opt_seq_cost([start_txn])))
        
        # Keep only best beam_width to start
        beam.sort(key=lambda x: x[2])
        beam = beam[:beam_width]
        
        # Iteratively expand beam
        for step in range(workload.num_txns - 1):
            candidates = []
            
            # Adaptive sampling: aggressive early exploration, focused later
            progress = step / (workload.num_txns - 1)
            if progress < 0.3:
                num_samples = min(35, len(beam[0][1]))
            elif progress < 0.6:
                num_samples = min(25, len(beam[0][1]))
            else:
                num_samples = min(15, len(beam[0][1]))
            
            for txn_seq, remaining, _ in beam:
                # Sample or use all remaining transactions
                if len(remaining) <= num_samples:
                    to_test = remaining
                else:
                    # Intelligent sampling: score a larger sample, then pick best
                    sample_size = min(num_samples * 2, len(remaining))
                    sampled = random.sample(remaining, sample_size)
                    scored_samples = [(t, compute_conflict_score(txn_seq, t)) for t in sampled]
                    scored_samples.sort(key=lambda x: x[1])
                    to_test = [t for t, _ in scored_samples[:num_samples]]
                
                for t in to_test:
                    new_seq = txn_seq + [t]
                    new_remaining = [x for x in remaining if x != t]
                    score = compute_conflict_score(txn_seq, t)
                    candidates.append((new_seq, new_remaining, score))
            
            # Keep best beam_width candidates with diversity injection
            candidates.sort(key=lambda x: x[2])
            
            # Take mostly best candidates, but add some diversity early on
            if step < workload.num_txns - 2 and len(candidates) > beam_width * 2:
                # 80% best, 20% diverse (from top 2*beam_width)
                num_best = int(beam_width * 0.8)
                beam = candidates[:num_best]
                
                # Add diverse candidates from remaining good options
                diverse_pool = candidates[num_best:beam_width * 2]
                if diverse_pool:
                    num_diverse = beam_width - num_best
                    beam.extend(random.sample(diverse_pool, min(num_diverse, len(diverse_pool))))
            else:
                beam = candidates[:beam_width]
        
        # Return best complete schedule
        best = min(beam, key=lambda x: x[2])
        final_cost = workload.get_opt_seq_cost(best[0])
        return final_cost, best[0]
    
    # Try different beam configurations
    best_cost = float('inf')
    best_schedule = None
    
    # Strategy 1: Wider beam search for better exploration
    # Use wider beams which have shown better results
    for beam_width in [12, 9, 7]:
        for _ in range(2):  # Multiple runs per width
            cost, schedule = beam_search_schedule(beam_width=beam_width)
            if cost < best_cost:
                best_cost = cost
                best_schedule = schedule
    
    # Strategy 2: Pure greedy as fallback (for small workloads)
    if workload.num_txns <= 15:
        for start in range(workload.num_txns):
            txn_seq = [start]
            remaining = [x for x in range(workload.num_txns) if x != start]
            
            for _ in range(workload.num_txns - 1):
                min_cost = float('inf')
                min_txn = -1
                
                for t in remaining:
                    test_seq = txn_seq + [t]
                    cost = workload.get_opt_seq_cost(test_seq)
                    if cost < min_cost:
                        min_cost = cost
                        min_txn = t
                
                txn_seq.append(min_txn)
                remaining.remove(min_txn)
            
            cost = workload.get_opt_seq_cost(txn_seq)
            if cost < best_cost:
                best_cost = cost
                best_schedule = txn_seq
    
    return best_cost, best_schedule

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
