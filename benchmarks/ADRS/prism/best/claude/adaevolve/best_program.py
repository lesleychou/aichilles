GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Iterative load balancing with batch reassignment using Hungarian algorithm.
    
    Strategy:
    1. Start with a simple first-fit placement
    2. Iteratively identify high-KVPR and low-KVPR GPUs
    3. Pool models from these GPUs
    4. Use linear_sum_assignment to optimally reassign models
    5. Repeat until convergence or max iterations
    
    This approach escapes local optima better than single-model moves by
    performing batch reassignments that consider global optimization.
    
    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs with minimized maximum KVPR
    """
    from scipy.optimize import linear_sum_assignment
    
    class GPUState:
        """Efficiently track GPU state for fast KVPR calculation."""
        def __init__(self):
            self.models = []
            self.used_mem = 0.0
            self.weighted_req = 0.0
        
        def add_model(self, model):
            self.models.append(model)
            self.used_mem += model.model_size
            self.weighted_req += model.req_rate / model.slo
        
        def remove_model(self, model):
            self.models.remove(model)
            self.used_mem -= model.model_size
            self.weighted_req -= model.req_rate / model.slo
        
        def get_kvpr(self):
            if self.used_mem >= GPU_MEM_SIZE:
                return float('inf')
            if not self.models:
                return 0.0
            return self.weighted_req / (GPU_MEM_SIZE - self.used_mem)
        
        def can_fit(self, model):
            return self.used_mem + model.model_size <= GPU_MEM_SIZE
    
    def initial_placement():
        """Create initial placement using first-fit by pressure."""
        gpu_states = [GPUState() for _ in range(gpu_num)]
        sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
        
        for model in sorted_models:
            placed = False
            for gpu_id in range(gpu_num):
                if gpu_states[gpu_id].can_fit(model):
                    gpu_states[gpu_id].add_model(model)
                    placed = True
                    break
            
            if not placed:
                raise ValueError(f"Unable to place model of size {model.model_size} GB")
        
        return gpu_states
    
    def iterative_reassignment(gpu_states, max_iter=15):
        """Iteratively reassign batches of models using Hungarian algorithm."""
        for iteration in range(max_iter):
            kvprs = [state.get_kvpr() for state in gpu_states]
            max_kvpr_before = max(kvprs)
            
            # Determine batch size (k GPUs from high, k from low)
            k = max(1, gpu_num // 2)
            
            # Identify high and low KVPR GPUs
            gpu_order = sorted(range(gpu_num), key=lambda i: kvprs[i])
            low_gpus = gpu_order[:k]
            high_gpus = gpu_order[-k:]
            
            # Pool models from these GPUs
            target_gpus = sorted(set(low_gpus + high_gpus))
            pooled_models = []
            
            for gpu_id in target_gpus:
                pooled_models.extend(gpu_states[gpu_id].models)
                # Remove all models from these GPUs
                for model in list(gpu_states[gpu_id].models):
                    gpu_states[gpu_id].remove_model(model)
            
            if not pooled_models:
                break
            
            # Build cost matrix: cost[model_idx, gpu_idx] = KVPR after adding model
            n_models = len(pooled_models)
            n_gpus = len(target_gpus)
            
            # We need to handle n_models != n_gpus by padding
            max_dim = max(n_models, n_gpus)
            cost_matrix = [[1e9 for _ in range(max_dim)] for _ in range(max_dim)]
            
            # Cache current state for each GPU
            gpu_cache = {}
            for idx, gpu_id in enumerate(target_gpus):
                gpu_cache[idx] = {
                    'used_mem': gpu_states[gpu_id].used_mem,
                    'weighted_req': gpu_states[gpu_id].weighted_req
                }
            
            # Fill cost matrix
            for m_idx, model in enumerate(pooled_models):
                for g_idx, gpu_id in enumerate(target_gpus):
                    # Check if model fits
                    if gpu_cache[g_idx]['used_mem'] + model.model_size > GPU_MEM_SIZE:
                        cost_matrix[m_idx][g_idx] = 1e9
                    else:
                        # Calculate KVPR after adding this model
                        new_mem = gpu_cache[g_idx]['used_mem'] + model.model_size
                        new_req = gpu_cache[g_idx]['weighted_req'] + model.req_rate / model.slo
                        remaining = GPU_MEM_SIZE - new_mem
                        
                        if remaining > 0:
                            kvpr_after = new_req / remaining
                            cost_matrix[m_idx][g_idx] = kvpr_after
                        else:
                            cost_matrix[m_idx][g_idx] = 1e9
            
            # Use Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Apply assignment
            for m_idx, g_idx in zip(row_ind, col_ind):
                if m_idx < n_models and g_idx < n_gpus:
                    model = pooled_models[m_idx]
                    gpu_id = target_gpus[g_idx]
                    
                    # Verify it fits before adding
                    if gpu_states[gpu_id].can_fit(model):
                        gpu_states[gpu_id].add_model(model)
                    else:
                        # Fallback: try other GPUs if assignment is infeasible
                        placed = False
                        for fallback_gpu in target_gpus:
                            if gpu_states[fallback_gpu].can_fit(model):
                                gpu_states[fallback_gpu].add_model(model)
                                placed = True
                                break
                        
                        if not placed:
                            # Try all GPUs as last resort
                            for fallback_gpu in range(gpu_num):
                                if gpu_states[fallback_gpu].can_fit(model):
                                    gpu_states[fallback_gpu].add_model(model)
                                    placed = True
                                    break
                            
                            if not placed:
                                raise ValueError(f"Unable to place model during reassignment")
            
            # Check for improvement
            new_kvprs = [state.get_kvpr() for state in gpu_states]
            max_kvpr_after = max(new_kvprs)
            
            # Early termination if no improvement
            if max_kvpr_after >= max_kvpr_before - 1e-9:
                break
        
        return gpu_states
    
    # Initialize placement
    gpu_states = initial_placement()
    
    # Apply iterative reassignment
    gpu_states = iterative_reassignment(gpu_states, max_iter=15)
    
    # Convert GPUState objects back to placement dictionary
    placement = {i: state.models for i, state in enumerate(gpu_states)}
    return placement

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
