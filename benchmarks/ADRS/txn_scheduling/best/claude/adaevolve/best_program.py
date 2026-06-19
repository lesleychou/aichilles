import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Simulated annealing with scipy.optimize.dual_annealing for global optimization.
    
    Key innovation: Uses dual annealing to explore the full permutation space globally,
    accepting uphill moves probabilistically to escape local minima that trap beam search.
    
    Approach:
    1. Encode schedules as continuous priority values in [0, num_txns)
    2. Use argsort to convert priorities to transaction permutations
    3. Cost function directly evaluates workload.get_opt_seq_cost
    4. Multiple restarts with dual_annealing for robustness
    5. Initialize with greedy heuristic for warm start
    
    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    from scipy.optimize import dual_annealing
    import numpy as np
    
    n = workload.num_txns
    
    def get_greedy_initial():
        """Generate a greedy initial solution based on conflict analysis."""
        # Analyze write conflicts
        write_counts = []
        for i in range(n):
            writes = sum(1 for op in workload.txns[i] if op[0] == 'w')
            write_counts.append((i, writes))
        
        # Sort by write count (fewer writes first to reduce conflicts)
        write_counts.sort(key=lambda x: x[1])
        greedy_order = [t for t, _ in write_counts]
        
        # Convert to priority encoding
        priorities = np.zeros(n)
        for rank, txn_id in enumerate(greedy_order):
            priorities[txn_id] = rank
        return priorities
    
    def priorities_to_schedule(priorities):
        """Convert continuous priority values to transaction schedule (permutation)."""
        # argsort gives indices that would sort the array
        # Lower priority value = earlier in schedule
        return list(np.argsort(priorities))
    
    def objective(priorities):
        """
        Objective function for dual_annealing.
        Converts priorities to schedule and evaluates makespan.
        """
        schedule = priorities_to_schedule(priorities)
        cost = workload.get_opt_seq_cost(schedule)
        return cost
    
    # Set up bounds: each priority can be anywhere in [0, n)
    bounds = [(0, n) for _ in range(n)]
    
    best_cost = float('inf')
    best_schedule = None
    
    # Number of restarts based on problem size and budget
    if n <= 5:
        restarts = max(3, num_seqs // 3)
        maxiter = 800
    elif n <= 10:
        restarts = max(2, num_seqs // 4)
        maxiter = 500
    else:
        restarts = max(2, num_seqs // 5)
        maxiter = 300
    
    for restart in range(restarts):
        # Vary initial guess for diversity
        if restart == 0:
            # Use greedy initialization for first run
            x0 = get_greedy_initial()
        elif restart == 1:
            # Random initialization
            x0 = np.random.uniform(0, n, n)
        else:
            # Perturbed greedy initialization
            x0 = get_greedy_initial()
            # Add noise to escape local minima
            noise = np.random.normal(0, n * 0.2, n)
            x0 = np.clip(x0 + noise, 0, n - 1e-6)
        
        try:
            # Run dual annealing
            # maxiter controls total iterations
            # initial_temp and restart_temp_ratio control annealing schedule
            # no_local_search=False enables local search refinement
            result = dual_annealing(
                objective,
                bounds=bounds,
                x0=x0,
                maxiter=maxiter,
                initial_temp=5230.0,  # Default good value
                restart_temp_ratio=2e-5,  # Default good value
                no_local_search=False,  # Enable local search
                seed=restart  # Different seed per restart
            )
            
            # Extract best solution
            final_schedule = priorities_to_schedule(result.x)
            final_cost = workload.get_opt_seq_cost(final_schedule)
            
            if final_cost < best_cost:
                best_cost = final_cost
                best_schedule = final_schedule
                
        except Exception as e:
            # Fallback to greedy if optimization fails
            greedy_priorities = get_greedy_initial()
            fallback_schedule = priorities_to_schedule(greedy_priorities)
            fallback_cost = workload.get_opt_seq_cost(fallback_schedule)
            
            if fallback_cost < best_cost:
                best_cost = fallback_cost
                best_schedule = fallback_schedule
    
    # Additional quick local refinement on best found
    if best_schedule is not None:
        # Try a few adjacent swaps
        for i in range(min(10, len(best_schedule) - 1)):
            test = best_schedule[:]
            test[i], test[i+1] = test[i+1], test[i]
            cost = workload.get_opt_seq_cost(test)
            if cost < best_cost:
                best_cost = cost
                best_schedule = test
    
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
