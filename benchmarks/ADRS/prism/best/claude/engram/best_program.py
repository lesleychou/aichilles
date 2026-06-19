GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Try multiple greedy placement strategies and keep the best result.
    Strategies: load-balance, capacity-aware, weighted-priority with fixed weights
    """

    # Helper function to calculate current max KVPR
    def calc_max_kvpr(model_size_used, weighted_req_rate, gpu_num):
        max_kvpr = 0.0
        for gid in range(gpu_num):
            remaining = GPU_MEM_SIZE - model_size_used[gid]
            if remaining > 0:
                kvpr = weighted_req_rate[gid] / remaining
                max_kvpr = max(max_kvpr, kvpr)
        return max_kvpr
    
    # Helper to calculate sum of squared KVPRs (for balanced placement)
    def calc_sum_squared_kvpr(model_size_used, weighted_req_rate, gpu_num):
        sum_sq = 0.0
        for gid in range(gpu_num):
            remaining = GPU_MEM_SIZE - model_size_used[gid]
            if remaining > 0:
                kvpr = weighted_req_rate[gid] / remaining
                sum_sq += kvpr ** 2
        return sum_sq
    
    # Helper to calculate max squared KVPR
    def calc_max_squared_kvpr(model_size_used, weighted_req_rate, gpu_num):
        max_sq = 0.0
        for gid in range(gpu_num):
            remaining = GPU_MEM_SIZE - model_size_used[gid]
            if remaining > 0:
                kvpr = weighted_req_rate[gid] / remaining
                max_sq = max(max_sq, kvpr ** 2)
        return max_sq

    def apply_local_search(placement, model_size_used, weighted_req_rate, gpu_num, intensity="high"):
        """Apply local search with tunable intensity"""
        
        # Set passes based on intensity
        if intensity == "low":
            phase1_passes, phase2_passes, phase3_attempts = 10, 5, 1
            top_k_1, top_k_2 = 3, 5
        elif intensity == "medium":
            phase1_passes, phase2_passes, phase3_attempts = 15, 8, 2
            top_k_1, top_k_2 = 4, 7
        else:  # high (ultra-aggressive for multi-strategy)
            phase1_passes, phase2_passes, phase3_attempts = 25, 15, 4
            top_k_1, top_k_2 = 6, 12
        
        # Phase 1: Pairwise swaps
        improved = True
        passes = 0
        while improved and passes < phase1_passes:
            passes += 1
            improved = False
            
            for gpu_id_1 in range(gpu_num):
                if not placement[gpu_id_1]:
                    continue
                for gpu_id_2 in range(gpu_id_1 + 1, gpu_num):
                    if not placement[gpu_id_2]:
                        continue
                    
                    models_1_sorted = sorted(placement[gpu_id_1], key=lambda m: m.req_rate / m.slo, reverse=True)
                    models_2_sorted = sorted(placement[gpu_id_2], key=lambda m: m.req_rate / m.slo, reverse=True)
                    
                    for model_1 in models_1_sorted[:top_k_1]:
                        for model_2 in models_2_sorted[:top_k_1]:
                            mem_needed_gpu2 = model_size_used[gpu_id_2] - model_2.model_size + model_1.model_size
                            mem_needed_gpu1 = model_size_used[gpu_id_1] - model_1.model_size + model_2.model_size
                            
                            if mem_needed_gpu1 <= GPU_MEM_SIZE and mem_needed_gpu2 <= GPU_MEM_SIZE:
                                current_max_kvpr = calc_max_kvpr(model_size_used, weighted_req_rate, gpu_num)
                                
                                new_weighted_req_rate_1 = weighted_req_rate[gpu_id_1] - model_1.req_rate / model_1.slo + model_2.req_rate / model_2.slo
                                new_weighted_req_rate_2 = weighted_req_rate[gpu_id_2] - model_2.req_rate / model_2.slo + model_1.req_rate / model_1.slo
                                
                                new_max_kvpr = 0.0
                                for gid in range(gpu_num):
                                    if gid == gpu_id_1:
                                        new_remaining = GPU_MEM_SIZE - mem_needed_gpu1
                                        new_kvpr = new_weighted_req_rate_1 / new_remaining if new_remaining > 0 else float('inf')
                                    elif gid == gpu_id_2:
                                        new_remaining = GPU_MEM_SIZE - mem_needed_gpu2
                                        new_kvpr = new_weighted_req_rate_2 / new_remaining if new_remaining > 0 else float('inf')
                                    else:
                                        new_remaining = GPU_MEM_SIZE - model_size_used[gid]
                                        new_kvpr = weighted_req_rate[gid] / new_remaining if new_remaining > 0 else float('inf')
                                    new_max_kvpr = max(new_max_kvpr, new_kvpr)
                                
                                if new_max_kvpr < current_max_kvpr - 1e-9:
                                    placement[gpu_id_1].remove(model_1)
                                    placement[gpu_id_2].remove(model_2)
                                    placement[gpu_id_1].append(model_2)
                                    placement[gpu_id_2].append(model_1)
                                    
                                    weighted_req_rate[gpu_id_1] = new_weighted_req_rate_1
                                    weighted_req_rate[gpu_id_2] = new_weighted_req_rate_2
                                    model_size_used[gpu_id_1] = mem_needed_gpu1
                                    model_size_used[gpu_id_2] = mem_needed_gpu2
                                    
                                    improved = True
                                    break
                        
                        if improved:
                            break

        # Phase 2: Single model moves
        improved = True
        passes = 0
        while improved and passes < phase2_passes:
            passes += 1
            improved = False
            
            for gpu_id_src in range(gpu_num):
                if not placement[gpu_id_src]:
                    continue
                
                models_to_try = sorted(placement[gpu_id_src], key=lambda m: m.req_rate / m.slo, reverse=True)
                
                for model_to_move in models_to_try[:top_k_2]:
                    for gpu_id_dst in range(gpu_num):
                        if gpu_id_dst == gpu_id_src:
                            continue
                        
                        mem_needed_dst = model_size_used[gpu_id_dst] + model_to_move.model_size
                        if mem_needed_dst > GPU_MEM_SIZE:
                            continue
                        
                        current_max_kvpr = calc_max_kvpr(model_size_used, weighted_req_rate, gpu_num)
                        
                        new_weighted_req_rate_src = weighted_req_rate[gpu_id_src] - model_to_move.req_rate / model_to_move.slo
                        new_weighted_req_rate_dst = weighted_req_rate[gpu_id_dst] + model_to_move.req_rate / model_to_move.slo
                        mem_needed_src = model_size_used[gpu_id_src] - model_to_move.model_size
                        
                        new_max_kvpr = 0.0
                        for gid in range(gpu_num):
                            if gid == gpu_id_src:
                                new_remaining = GPU_MEM_SIZE - mem_needed_src
                                new_kvpr = new_weighted_req_rate_src / new_remaining if new_remaining > 0 else 0.0
                            elif gid == gpu_id_dst:
                                new_remaining = GPU_MEM_SIZE - mem_needed_dst
                                new_kvpr = new_weighted_req_rate_dst / new_remaining if new_remaining > 0 else float('inf')
                            else:
                                new_remaining = GPU_MEM_SIZE - model_size_used[gid]
                                new_kvpr = weighted_req_rate[gid] / new_remaining if new_remaining > 0 else float('inf')
                            new_max_kvpr = max(new_max_kvpr, new_kvpr)
                        
                        if new_max_kvpr < current_max_kvpr - 1e-9:
                            placement[gpu_id_src].remove(model_to_move)
                            placement[gpu_id_dst].append(model_to_move)
                            
                            weighted_req_rate[gpu_id_src] = new_weighted_req_rate_src
                            weighted_req_rate[gpu_id_dst] = new_weighted_req_rate_dst
                            model_size_used[gpu_id_src] = mem_needed_src
                            model_size_used[gpu_id_dst] = mem_needed_dst
                            
                            improved = True
                            break
                    
                    if improved:
                        break
        
        # Phase 3: Chain moves (try moving model from A->B after moving another from B)
        for attempt in range(phase3_attempts):
            improved = False
            for gpu_id_1 in range(gpu_num):
                if not placement[gpu_id_1] or len(placement[gpu_id_1]) == 0:
                    continue
                for gpu_id_2 in range(gpu_num):
                    if gpu_id_2 == gpu_id_1 or not placement[gpu_id_2]:
                        continue
                    
                    # Try moving top model from gpu_id_2 to gpu_id_1, then move top from gpu_id_1 to gpu_id_2
                    models_2 = sorted(placement[gpu_id_2], key=lambda m: m.req_rate / m.slo, reverse=True)
                    if len(models_2) == 0:
                        continue
                    
                    model_2_to_move = models_2[0]
                    mem_after_remove = model_size_used[gpu_id_1] + model_2_to_move.model_size
                    if mem_after_remove > GPU_MEM_SIZE:
                        continue
                    
                    models_1 = sorted(placement[gpu_id_1], key=lambda m: m.req_rate / m.slo, reverse=True)
                    for model_1_to_move in models_1[:2]:
                        mem_after_remove_1 = model_size_used[gpu_id_2] - model_2_to_move.model_size + model_1_to_move.model_size
                        if mem_after_remove_1 > GPU_MEM_SIZE:
                            continue
                        
                        current_max_kvpr = calc_max_kvpr(model_size_used, weighted_req_rate, gpu_num)
                        
                        new_wr_1 = weighted_req_rate[gpu_id_1] - model_1_to_move.req_rate / model_1_to_move.slo + model_2_to_move.req_rate / model_2_to_move.slo
                        new_wr_2 = weighted_req_rate[gpu_id_2] - model_2_to_move.req_rate / model_2_to_move.slo + model_1_to_move.req_rate / model_1_to_move.slo
                        
                        new_max_kvpr = 0.0
                        for gid in range(gpu_num):
                            if gid == gpu_id_1:
                                remaining = GPU_MEM_SIZE - mem_after_remove
                                kvpr = new_wr_1 / remaining if remaining > 0 else float('inf')
                            elif gid == gpu_id_2:
                                remaining = GPU_MEM_SIZE - mem_after_remove_1
                                kvpr = new_wr_2 / remaining if remaining > 0 else float('inf')
                            else:
                                remaining = GPU_MEM_SIZE - model_size_used[gid]
                                kvpr = weighted_req_rate[gid] / remaining if remaining > 0 else float('inf')
                            new_max_kvpr = max(new_max_kvpr, kvpr)
                        
                        if new_max_kvpr < current_max_kvpr - 1e-9:
                            placement[gpu_id_1].remove(model_1_to_move)
                            placement[gpu_id_2].remove(model_2_to_move)
                            placement[gpu_id_1].append(model_2_to_move)
                            placement[gpu_id_2].append(model_1_to_move)
                            
                            weighted_req_rate[gpu_id_1] = new_wr_1
                            weighted_req_rate[gpu_id_2] = new_wr_2
                            model_size_used[gpu_id_1] = mem_after_remove
                            model_size_used[gpu_id_2] = mem_after_remove_1
                            
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                break

    def solve_with_strategy(strategy_name, placement_rule="lowest_kvpr"):
        """Solve with a specific placement strategy and placement rule"""
        
        # Prepare sorted models based on strategy
        if strategy_name == "load_balance":
            sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
        elif strategy_name == "capacity_aware":
            sorted_models = sorted(models, key=lambda m: m.model_size, reverse=True)
        elif strategy_name == "weighted_priority":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo + 0.042 * m.model_size), reverse=True)
        elif strategy_name == "weighted_alt1":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo + 0.030 * m.model_size), reverse=True)
        elif strategy_name == "weighted_alt2":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo + 0.055 * m.model_size), reverse=True)
        elif strategy_name == "slo_capacity":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo * m.model_size), reverse=True)
        elif strategy_name == "weighted_alt3":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo + 0.025 * m.model_size), reverse=True)
        elif strategy_name == "weighted_alt4":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo + 0.060 * m.model_size), reverse=True)
        elif strategy_name == "inverse_capacity":
            sorted_models = sorted(models, key=lambda m: m.model_size, reverse=False)
        elif strategy_name == "low_priority":
            sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=False)
        elif strategy_name == "high_slo_first":
            sorted_models = sorted(models, key=lambda m: m.slo, reverse=True)
        elif strategy_name == "low_slo_first":
            sorted_models = sorted(models, key=lambda m: m.slo, reverse=False)
        elif strategy_name == "inverse_weighted1":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo + 0.042 * m.model_size), reverse=False)
        elif strategy_name == "size_biased":
            sorted_models = sorted(models, key=lambda m: (0.7 * (m.req_rate / m.slo) + 0.3 * (m.model_size / max(m.model_size for m in models) if models else 1)), reverse=True)
        elif strategy_name == "req_rate_only":
            sorted_models = sorted(models, key=lambda m: m.req_rate, reverse=True)
        elif strategy_name == "req_rate_low":
            sorted_models = sorted(models, key=lambda m: m.req_rate, reverse=False)
        elif strategy_name == "slo_weighted":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / (m.slo ** 0.5)), reverse=True)
        elif strategy_name == "weighted_015":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo + 0.015 * m.model_size), reverse=True)
        elif strategy_name == "weighted_020":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo + 0.020 * m.model_size), reverse=True)
        elif strategy_name == "product_priority_size":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo) * m.model_size, reverse=True)
        elif strategy_name == "normalized_combo":
            max_size = max(m.model_size for m in models) if models else 1
            max_priority = max(m.req_rate / m.slo for m in models) if models else 1
            sorted_models = sorted(models, key=lambda m: ((m.req_rate / m.slo) / max_priority + (m.model_size / max_size)), reverse=True)
        elif strategy_name == "sqrt_product":
            sorted_models = sorted(models, key=lambda m: ((m.req_rate / m.slo) * m.model_size) ** 0.5, reverse=True)
        elif strategy_name == "cubic_priority":
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo) ** 1.5, reverse=True)
        elif strategy_name == "cubic_size":
            sorted_models = sorted(models, key=lambda m: m.model_size ** 1.5, reverse=True)
        else:
            return None, float('inf')

        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        model_size_used = [0.0 for _ in range(gpu_num)]
        weighted_req_rate = [0.0 for _ in range(gpu_num)]

        # Greedy placement with chosen rule
        for model in sorted_models:
            best_idx = None
            best_value = float('inf')

            for gpu_id in range(gpu_num):
                available_mem = GPU_MEM_SIZE - model_size_used[gpu_id]
                
                if model.model_size <= available_mem:
                    new_weighted_req_rate = weighted_req_rate[gpu_id] + model.req_rate / model.slo
                    remaining_mem = available_mem - model.model_size
                    
                    if remaining_mem > 0:
                        if placement_rule == "lowest_kvpr":
                            # Original: place on GPU with lowest KVPR
                            value = new_weighted_req_rate / remaining_mem
                        elif placement_rule == "balanced":
                            # Alternative: place to minimize sum of squared KVPRs
                            temp_size = model_size_used[gpu_id]
                            temp_rate = weighted_req_rate[gpu_id]
                            model_size_used[gpu_id] += model.model_size
                            weighted_req_rate[gpu_id] = new_weighted_req_rate
                            value = calc_sum_squared_kvpr(model_size_used, weighted_req_rate, gpu_num)
                            model_size_used[gpu_id] = temp_size
                            weighted_req_rate[gpu_id] = temp_rate
                        elif placement_rule == "penalize_extremes":
                            # Penalize extreme values: minimize max squared KVPR
                            temp_size = model_size_used[gpu_id]
                            temp_rate = weighted_req_rate[gpu_id]
                            model_size_used[gpu_id] += model.model_size
                            weighted_req_rate[gpu_id] = new_weighted_req_rate
                            value = calc_max_squared_kvpr(model_size_used, weighted_req_rate, gpu_num)
                            model_size_used[gpu_id] = temp_size
                            weighted_req_rate[gpu_id] = temp_rate
                        else:
                            value = new_weighted_req_rate / remaining_mem
                    else:
                        value = float('inf')
                    
                    if value < best_value:
                        best_value = value
                        best_idx = gpu_id

            if best_idx is None:
                raise ValueError(f"Unable to place model")

            placement[best_idx].append(model)
            weighted_req_rate[best_idx] += model.req_rate / model.slo
            model_size_used[best_idx] += model.model_size

        # Apply local search with high intensity for all
        apply_local_search(placement, model_size_used, weighted_req_rate, gpu_num, intensity="high")

        # Final refinement: ultra-aggressive refinement on bottleneck GPU
        for final_round in range(2):
            # Find bottleneck GPU
            max_kvpr_val = 0.0
            bottleneck_gpu = -1
            for gid in range(gpu_num):
                remaining = GPU_MEM_SIZE - model_size_used[gid]
                if remaining > 0:
                    kvpr = weighted_req_rate[gid] / remaining
                    if kvpr > max_kvpr_val:
                        max_kvpr_val = kvpr
                        bottleneck_gpu = gid
            
            if bottleneck_gpu == -1:
                break
            
            # Try to move expensive models from bottleneck to underutilized GPUs
            improved = False
            models_on_bottleneck = sorted(placement[bottleneck_gpu], key=lambda m: m.req_rate / m.slo, reverse=True)
            
            for model_to_move in models_on_bottleneck[:3]:
                for dst_gpu in range(gpu_num):
                    if dst_gpu == bottleneck_gpu:
                        continue
                    
                    if model_size_used[dst_gpu] + model_to_move.model_size > GPU_MEM_SIZE:
                        continue
                    
                    current_max = calc_max_kvpr(model_size_used, weighted_req_rate, gpu_num)
                    
                    new_wr_src = weighted_req_rate[bottleneck_gpu] - model_to_move.req_rate / model_to_move.slo
                    new_wr_dst = weighted_req_rate[dst_gpu] + model_to_move.req_rate / model_to_move.slo
                    new_mem_src = model_size_used[bottleneck_gpu] - model_to_move.model_size
                    new_mem_dst = model_size_used[dst_gpu] + model_to_move.model_size
                    
                    new_max = 0.0
                    for gid in range(gpu_num):
                        if gid == bottleneck_gpu:
                            remaining = GPU_MEM_SIZE - new_mem_src
                            kvpr = new_wr_src / remaining if remaining > 0 else 0.0
                        elif gid == dst_gpu:
                            remaining = GPU_MEM_SIZE - new_mem_dst
                            kvpr = new_wr_dst / remaining if remaining > 0 else float('inf')
                        else:
                            remaining = GPU_MEM_SIZE - model_size_used[gid]
                            kvpr = weighted_req_rate[gid] / remaining if remaining > 0 else float('inf')
                        new_max = max(new_max, kvpr)
                    
                    if new_max < current_max - 1e-9:
                        placement[bottleneck_gpu].remove(model_to_move)
                        placement[dst_gpu].append(model_to_move)
                        weighted_req_rate[bottleneck_gpu] = new_wr_src
                        weighted_req_rate[dst_gpu] = new_wr_dst
                        model_size_used[bottleneck_gpu] = new_mem_src
                        model_size_used[dst_gpu] = new_mem_dst
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break

        # Calculate final max KVPR
        max_kvpr = calc_max_kvpr(model_size_used, weighted_req_rate, gpu_num)
        return placement, max_kvpr

    # Try all strategies with both placement rules
    best_placement = None
    best_max_kvpr = float('inf')
    
    strategies = ["load_balance", "capacity_aware", "weighted_priority", "weighted_alt1", "weighted_alt2", "weighted_alt3", "weighted_alt4", "slo_capacity", "inverse_capacity", "low_priority", "high_slo_first", "low_slo_first", "inverse_weighted1", "size_biased", "req_rate_only", "req_rate_low", "slo_weighted", "weighted_015", "weighted_020", "product_priority_size", "normalized_combo", "sqrt_product", "cubic_priority", "cubic_size"]
    
    # Try with lowest_kvpr placement (original, faster)
    for strategy in strategies:
        try:
            placement, max_kvpr = solve_with_strategy(strategy, placement_rule="lowest_kvpr")
            if placement is not None and max_kvpr < best_max_kvpr:
                best_max_kvpr = max_kvpr
                best_placement = placement
        except:
            continue
    
    # Try all strategies with balanced placement
    for strategy in strategies:
        try:
            placement, max_kvpr = solve_with_strategy(strategy, placement_rule="balanced")
            if placement is not None and max_kvpr < best_max_kvpr:
                best_max_kvpr = max_kvpr
                best_placement = placement
        except:
            continue
    
    # Try all strategies with penalize_extremes placement
    for strategy in strategies:
        try:
            placement, max_kvpr = solve_with_strategy(strategy, placement_rule="penalize_extremes")
            if placement is not None and max_kvpr < best_max_kvpr:
                best_max_kvpr = max_kvpr
                best_placement = placement
        except:
            continue
    
    if best_placement is None:
        # Fallback
        best_placement, _ = solve_with_strategy("load_balance")
    
    return best_placement

# EVOLVE-BLOCK-END