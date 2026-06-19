GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # Sort by a combined metric: (req_rate/slo) * model_size to prioritize "difficult" models
    # This helps place constrained models first when we have more flexibility
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo) * m.model_size, reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # sum of r_j / s_j per GPU

    # 3) Assign each model to the GPU that minimizes KVPR after placement
    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')

        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id]:
                # Calculate what KVPR would be AFTER placing this model
                new_mem = shared_kv[gpu_id] - model.model_size
                new_weighted = weighted_req_rate[gpu_id] + model.req_rate / model.slo
                
                # Avoid division by zero - if memory would be exhausted, use large penalty
                if new_mem > 0:
                    future_ratio = new_weighted / new_mem
                    
                    # Add small penalty for unbalanced memory usage to encourage spreading
                    # This helps avoid creating one overloaded GPU early
                    max_other_kvpr = max((weighted_req_rate[i] / shared_kv[i] if shared_kv[i] > 0 else 0) 
                                        for i in range(gpu_num) if i != gpu_id)
                    balance_factor = 1.0 + 0.1 * max(0, future_ratio - max_other_kvpr)
                    future_ratio *= balance_factor
                else:
                    future_ratio = float('inf')
                
                # Prefer this GPU if it has better KVPR, or similar KVPR but more memory
                if future_ratio < best_ratio or (abs(future_ratio - best_ratio) < 0.01 and new_mem > shared_kv[best_idx] - model.model_size):
                    best_ratio = future_ratio
                    best_idx = gpu_id

        # Failure: if no GPU can fit, raise an error instead of overcommitting
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size

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
