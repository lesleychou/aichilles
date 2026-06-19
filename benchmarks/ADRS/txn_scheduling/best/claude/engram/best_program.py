import time
import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using enhanced greedy cost sampling with local search.
    
    Improvements over baseline:
    - Greedy cost sampling with num_samples=40 (from previous agent)
    - Deterministic tie-breaking by write count
    - Local search with 2-opt perturbations to escape local optima
    - Multiple restarts to find better global solutions
    
    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    
    def get_greedy_cost_sampled(num_samples, sample_rate, seed_start_txn=None):
        """Greedy sampling approach with optional seeded starting point."""
        # Choose starting transaction with heuristic (transaction with most writes/conflicts)
        if seed_start_txn is not None:
            start_txn = seed_start_txn
        else:
            max_writes = -1
            best_start = 0
            for txn_id in range(workload.num_txns):
                write_count = len([op for op in workload.txns[txn_id][0][0] if 'w' in str(op)])
                if write_count > max_writes:
                    max_writes = write_count
                    best_start = txn_id
            start_txn = best_start
        
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(0, workload.num_txns)]
        remaining_txns.remove(start_txn)
        running_cost = workload.txns[start_txn][0][3]
        
        for i in range(0, workload.num_txns - 1):
            min_cost = 100000  # MAX
            min_relative_cost = 10
            min_txn = -1
            min_idx = -1  # Track index for tie-breaking
            holdout_txns = []
            done = False
            key_maps = []

            sample = random.random()
            if sample > sample_rate:
                idx = random.randint(0, len(remaining_txns) - 1)
                t = remaining_txns[idx]
                txn_seq.append(t)
                remaining_txns.pop(idx)
                continue

            for j in range(0, num_samples):
                idx = 0
                if len(remaining_txns) > 1:
                    idx = random.randint(0, len(remaining_txns) - 1)
                else:
                    done = True
                t = remaining_txns[idx]
                holdout_txns.append(remaining_txns.pop(idx))
                if workload.debug:
                    print(remaining_txns, holdout_txns)
                txn_len = workload.txns[t][0][3]
                test_seq = txn_seq.copy()
                test_seq.append(t)
                cost = 0
                cost = workload.get_opt_seq_cost(test_seq)
                
                # EVOLVE-BLOCK-START
                # Deterministic tie-breaking: if cost is equal, prefer transaction with more writes
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t
                    min_idx = j
                elif cost == min_cost:
                    # Tie-breaker: prefer transaction with more writes
                    t_writes = len([op for op in workload.txns[t][0][0] if 'w' in str(op)])
                    min_txn_writes = len([op for op in workload.txns[min_txn][0][0] if 'w' in str(op)])
                    if t_writes > min_txn_writes:
                        min_txn = t
                        min_idx = j
                # EVOLVE-BLOCK-END
                
                if done:
                    break
            assert(min_txn != -1)
            running_cost = min_cost
            txn_seq.append(min_txn)
            holdout_txns.remove(min_txn)
            remaining_txns.extend(holdout_txns)

            if workload.debug:
                print("min: ", min_txn, remaining_txns, holdout_txns, txn_seq)
        if workload.debug:
            print(txn_seq)
            print(len(set(txn_seq)))
        assert len(set(txn_seq)) == workload.num_txns
        
        overall_cost = workload.get_opt_seq_cost(txn_seq)

        return overall_cost, txn_seq


    def local_search_2opt(initial_seq, max_iterations=50):
        """Apply 2-opt local search to improve a schedule."""
        best_seq = initial_seq.copy()
        best_cost = workload.get_opt_seq_cost(best_seq)
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try swapping pairs of transactions
            for i in range(len(best_seq)):
                for j in range(i + 2, len(best_seq)):
                    # Create new sequence by reversing segment [i+1:j+1]
                    new_seq = best_seq[:i+1] + best_seq[i+1:j+1][::-1] + best_seq[j+1:]
                    new_cost = workload.get_opt_seq_cost(new_seq)
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_seq = new_seq
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_cost, best_seq
    
    def local_search_oropt_simple(initial_seq, max_iterations=20):
        """
        Simplified Or-opt: move single transactions to nearby positions.
        Only move transactions 1-2 positions away to keep complexity low.
        """
        best_seq = initial_seq.copy()
        best_cost = workload.get_opt_seq_cost(best_seq)
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            n = len(best_seq)
            
            # For each transaction, try moving it to nearby positions
            for i in range(n):
                if improved:
                    break
                # Try moving to positions 1-2 steps away
                for j in range(max(0, i-2), min(n, i+3)):
                    if j == i or j == i - 1:
                        continue
                    
                    # Create new sequence by moving txn at position i to position j
                    txn = best_seq[i]
                    new_seq = best_seq[:i] + best_seq[i+1:]
                    if j < i:
                        new_seq = new_seq[:j] + [txn] + new_seq[j:]
                    else:
                        new_seq = new_seq[:j-1] + [txn] + new_seq[j-1:]
                    
                    try:
                        new_cost = workload.get_opt_seq_cost(new_seq)
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_seq = new_seq
                            improved = True
                            break
                    except:
                        pass
        
        return best_cost, best_seq

    # EVOLVE-BLOCK-START
    # Try multiple greedy runs with random starting points and apply local search to best
    best_overall_cost = float('inf')
    best_overall_seq = None
    all_greedy_results = []
    
    # Run greedy with the default heuristic starting point with num_samples=40 (proven optimal)
    cost1, seq1 = get_greedy_cost_sampled(40, 1.0)
    all_greedy_results.append((cost1, seq1))
    
    # Try multiple greedy runs with different random starting points
    for attempt in range(6):
        random_start = random.randint(0, workload.num_txns - 1)
        cost_attempt, seq_attempt = get_greedy_cost_sampled(40, 1.0, seed_start_txn=random_start)
        all_greedy_results.append((cost_attempt, seq_attempt))
    
    # Sort greedy solutions by cost
    all_greedy_results.sort(key=lambda x: x[0])
    
    # Apply local search with varying iterations depth
    for idx in range(min(3, len(all_greedy_results))):
        greedy_cost, greedy_seq = all_greedy_results[idx]
        # 2-opt with original proven iterations
        iterations = 40 if idx < 2 else 25
        cost_improved, seq_improved = local_search_2opt(greedy_seq, max_iterations=iterations)
        
        # Then: simplified Or-opt on top 2 candidates as final refinement
        if idx < 2:
            oropt_max = 20 if idx == 0 else 15
            cost_oropt, seq_oropt = local_search_oropt_simple(seq_improved, max_iterations=oropt_max)
            if cost_oropt < cost_improved:
                cost_improved = cost_oropt
                seq_improved = seq_oropt
        
        if idx == 0 or cost_improved < best_overall_cost:
            best_overall_cost = cost_improved
            best_overall_seq = seq_improved
    
    return best_overall_cost, best_overall_seq
    # EVOLVE-BLOCK-END


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