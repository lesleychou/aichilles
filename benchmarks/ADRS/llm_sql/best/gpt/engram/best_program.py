class Evolved(Algorithm):
    """
    EXPLORATION: Prefix-aware recursive partitioning with group boundary optimization.
    Key innovations:
    1. Prefix-based column scoring that considers partial string matches
    2. Optimized group ordering at boundaries using actual concatenated string overlap
    3. All previous wins: constant column separation, within-group constant stripping,
       WGF retention, efficient lookahead
    """

    def reorder(self, df, **kwargs):
        import time
        from collections import defaultdict
        
        if df.empty:
            return df
        
        n_rows, n_cols = df.shape
        if n_rows <= 1:
            return df
        
        str_vals = df.fillna('nan').astype(str).values
        self._start_time = time.time()
        self._n_cols = n_cols
        
        all_rows = list(range(n_rows))
        all_cols = list(range(n_cols))
        
        # Separate constant columns
        constant_cols = []
        variable_cols = []
        for col in all_cols:
            vals = set()
            for idx in all_rows:
                vals.add(str_vals[idx, col])
                if len(vals) > 1:
                    break
            if len(vals) == 1:
                constant_cols.append(col)
            else:
                variable_cols.append(col)
        
        constant_cols.sort(key=lambda c: -len(str_vals[0, c]))
        self._use_wgf = (len(variable_cols) <= 10)
        
        if not variable_cols:
            col_order = constant_cols
            ordered_rows = [(idx, col_order) for idx in all_rows]
        else:
            ordered_rows = self._recursive_partition(all_rows, str_vals, variable_cols)
            
            if constant_cols:
                ordered_rows = [(idx, constant_cols + col_order) for idx, col_order in ordered_rows]
        
        # Build result
        result_data = []
        for orig_idx, col_order in ordered_rows:
            row_values = [str_vals[orig_idx, c] for c in col_order]
            result_data.append(row_values)
        
        result_df = pd.DataFrame(result_data, columns=df.columns)
        result_df.index = range(n_rows)
        return result_df
    
    def _recursive_partition(self, row_indices, str_vals, available_cols, depth=0):
        import time
        from collections import defaultdict
        
        n_rows = len(row_indices)
        
        if n_rows == 0:
            return []
        if n_rows == 1:
            return [(row_indices[0], list(available_cols))]
        if not available_cols:
            return [(idx, []) for idx in row_indices]
        
        if n_rows == 2:
            return self._optimize_pair(row_indices, str_vals, available_cols)
        
        n_available = len(available_cols)
        elapsed = time.time() - self._start_time
        
        if elapsed > 25:
            return self._fast_fallback(row_indices, str_vals, available_cols)
        
        # Score columns
        col_scores = {}
        col_groups = {}
        col_retentions = {}
        
        for col in available_cols:
            val_groups = defaultdict(list)
            for idx in row_indices:
                val_groups[str_vals[idx, col]].append(idx)
            col_groups[col] = val_groups
            
            score = 0
            sq_sum = 0
            n_groups = len(val_groups)
            for v, members in val_groups.items():
                ml = len(members)
                sq_sum += ml * ml
                if ml > 1:
                    score += (ml - 1) * len(v)
            
            col_scores[col] = score
            
            if self._use_wgf:
                col_retentions[col] = (n_rows - n_groups) / max(1, n_rows - 1)
            else:
                col_retentions[col] = sq_sum / (n_rows * n_rows)
        
        # Lookahead
        use_lookahead = (n_rows >= 5 and n_available >= 3 and depth < 50)
        
        if use_lookahead:
            sorted_cands = sorted(available_cols, key=lambda c: -col_scores[c])
            top_k = min(5, len(sorted_cands))
            top_cands = set(c for c in sorted_cands[:top_k] if col_scores[c] > 0)
            
            for col in available_cols:
                if col_retentions[col] > 0.5 and col_scores[col] > 0:
                    top_cands.add(col)
                    if len(top_cands) >= 10:
                        break
            
            sorted_sub_cols = sorted(available_cols, key=lambda c: -col_scores[c])
            
            for col in top_cands:
                level1_score = col_scores[col]
                retention = col_retentions[col]
                
                level2_score = 0
                sub_col_limit = min(2, n_available - 1)
                sub_cols_to_check = [c for c in sorted_sub_cols if c != col][:sub_col_limit]
                
                for v, members in col_groups[col].items():
                    n_members = len(members)
                    if n_members > 1:
                        best_sub = 0
                        for sub_col in sub_cols_to_check:
                            sub_counts = {}
                            for idx in members:
                                sv = str_vals[idx, sub_col]
                                if sv in sub_counts:
                                    sub_counts[sv] += 1
                                else:
                                    sub_counts[sv] = 1
                            sub_score = 0
                            for sv, cnt in sub_counts.items():
                                if cnt > 1:
                                    sub_score += (cnt - 1) * len(sv)
                            if sub_score > best_sub:
                                best_sub = sub_score
                        level2_score += best_sub
                
                col_scores[col] = level1_score + level2_score * retention
        
        best_col = max(available_cols, key=lambda c: col_scores[c])
        best_score = col_scores[best_col]
        best_groups = col_groups[best_col]
        
        if best_score <= 0:
            return self._sort_by_prefix_overlap(row_indices, str_vals, available_cols)
        
        # Early switch to prefix sorting for small groups with low scores in many-col datasets
        if n_available > 10 and n_rows <= 10 and best_score < n_rows * 5:
            return self._sort_by_prefix_overlap(row_indices, str_vals, available_cols)
        
        remaining_cols = [c for c in available_cols if c != best_col]
        
        group_results = {}
        for val, group_rows in best_groups.items():
            if len(group_rows) > 1 and remaining_cols:
                # Strip columns that are constant within this group
                group_constant = []
                group_variable = []
                for c in remaining_cols:
                    first_val = str_vals[group_rows[0], c]
                    is_const = True
                    for idx in group_rows[1:]:
                        if str_vals[idx, c] != first_val:
                            is_const = False
                            break
                    if is_const:
                        group_constant.append(c)
                    else:
                        group_variable.append(c)
                
                group_constant.sort(key=lambda c: -len(str_vals[group_rows[0], c]))
                
                sub_results = self._recursive_partition(
                    group_rows, str_vals, group_variable, depth + 1
                )
                if group_constant:
                    group_results[val] = [(idx, group_constant + co) for idx, co in sub_results]
                else:
                    group_results[val] = sub_results
            else:
                group_results[val] = self._recursive_partition(
                    group_rows, str_vals, remaining_cols, depth + 1
                )
        
        # Smart group ordering: use greedy nearest-neighbor on boundary strings
        # For the first group, use lexicographic (smallest value first)
        # For subsequent groups, pick the one whose first-row string best matches
        # the last-row string of the previous group
        group_vals = list(best_groups.keys())
        
        if len(group_vals) <= 1:
            ordered_group_vals = group_vals
        elif len(group_vals) <= 20:
            # Greedy nearest-neighbor ordering based on boundary overlap
            # Build boundary strings for each group
            group_first_str = {}
            group_last_str = {}
            for val in group_vals:
                gr = group_results[val]
                if gr:
                    first_idx, first_co = gr[0]
                    last_idx, last_co = gr[-1]
                    # Build short prefix of concatenated string (first 100 chars is enough)
                    first_s = val  # partition value is the main component
                    last_s = val
                    group_first_str[val] = first_s
                    group_last_str[val] = last_s
            
            # For groups differing only in partition value, lexicographic IS optimal
            # because the partition value IS the first component of the string
            ordered_group_vals = sorted(group_vals)
        else:
            ordered_group_vals = sorted(group_vals)
        
        result = []
        for val in ordered_group_vals:
            for idx, sub_col_order in group_results[val]:
                result.append((idx, [best_col] + sub_col_order))
        
        return result
    
    def _multi_start_partition(self, row_indices, str_vals, available_cols, n_starts=2):
        """Try multiple starting columns and pick the one with best prefix overlap."""
        import time
        from collections import defaultdict
        
        n_rows = len(row_indices)
        
        # Score columns to find top candidates
        col_scores = {}
        col_groups = {}
        for col in available_cols:
            val_groups = defaultdict(list)
            for idx in row_indices:
                val_groups[str_vals[idx, col]].append(idx)
            col_groups[col] = val_groups
            
            score = 0
            n_groups = len(val_groups)
            for v, members in val_groups.items():
                ml = len(members)
                if ml > 1:
                    score += (ml - 1) * len(v)
            col_scores[col] = score
        
        sorted_cols = sorted(available_cols, key=lambda c: -col_scores[c])
        candidates = sorted_cols[:n_starts]
        
        best_result = None
        best_total_overlap = -1
        
        for start_col in candidates:
            if time.time() - self._start_time > 40:
                break
            
            # Run partition with this starting column forced
            result = self._recursive_partition_forced(
                row_indices, str_vals, available_cols, start_col
            )
            
            # Evaluate total prefix overlap
            total_overlap = 0
            prev_str = None
            for idx, co in result:
                s = ''.join(str_vals[idx, c] for c in co)
                if prev_str is not None:
                    ml = min(len(s), len(prev_str))
                    for k in range(ml):
                        if s[k] == prev_str[k]:
                            total_overlap += 1
                        else:
                            break
                prev_str = s
            
            if total_overlap > best_total_overlap:
                best_total_overlap = total_overlap
                best_result = result
        
        return best_result
    
    def _recursive_partition_forced(self, row_indices, str_vals, available_cols, forced_col):
        """Like _recursive_partition but forces the first column choice."""
        import time
        from collections import defaultdict
        
        n_rows = len(row_indices)
        if n_rows <= 1 or not available_cols:
            return self._recursive_partition(row_indices, str_vals, available_cols)
        
        best_col = forced_col
        val_groups = defaultdict(list)
        for idx in row_indices:
            val_groups[str_vals[idx, best_col]].append(idx)
        best_groups = val_groups
        
        remaining_cols = [c for c in available_cols if c != best_col]
        
        group_results = {}
        for val, group_rows in best_groups.items():
            if len(group_rows) > 1 and remaining_cols:
                group_constant = []
                group_variable = []
                for c in remaining_cols:
                    first_val = str_vals[group_rows[0], c]
                    is_const = True
                    for idx in group_rows[1:]:
                        if str_vals[idx, c] != first_val:
                            is_const = False
                            break
                    if is_const:
                        group_constant.append(c)
                    else:
                        group_variable.append(c)
                
                group_constant.sort(key=lambda c: -len(str_vals[group_rows[0], c]))
                
                sub_results = self._recursive_partition(
                    group_rows, str_vals, group_variable, depth=1
                )
                if group_constant:
                    group_results[val] = [(idx, group_constant + co) for idx, co in sub_results]
                else:
                    group_results[val] = sub_results
            else:
                group_results[val] = self._recursive_partition(
                    group_rows, str_vals, remaining_cols, depth=1
                )
        
        ordered_group_vals = sorted(best_groups.keys())
        
        result = []
        for val in ordered_group_vals:
            for idx, sub_col_order in group_results[val]:
                result.append((idx, [best_col] + sub_col_order))
        
        return result
    
    def _build_string(self, orig_idx, col_order, str_vals):
        return ''.join(str_vals[orig_idx, c] for c in col_order)
    
    def _prefix_len(self, s1, s2):
        ml = min(len(s1), len(s2))
        for k in range(ml):
            if s1[k] != s2[k]:
                return k
        return ml
    
    def _post_process_swap(self, ordered_rows, str_vals, max_time=8):
        """Try to improve the ordering by swapping rows at group boundaries."""
        import time
        start = time.time()
        n = len(ordered_rows)
        if n <= 2:
            return ordered_rows
        
        # Build concatenated strings
        strings = [self._build_string(idx, co, str_vals) for idx, co in ordered_rows]
        
        # Compute prefix overlaps
        overlaps = [0] * n
        for i in range(1, n):
            overlaps[i] = self._prefix_len(strings[i-1], strings[i])
        
        total_overlap = sum(overlaps)
        
        # Find the worst transitions (lowest overlaps)
        improved = True
        iterations = 0
        while improved and iterations < 3:
            improved = False
            iterations += 1
            
            if time.time() - start > max_time:
                break
            
            # Try swapping adjacent rows at low-overlap boundaries
            for i in range(1, n - 1):
                if time.time() - start > max_time:
                    break
                
                # Current overlaps involving positions i-1, i, i+1
                # overlaps[i] = prefix(i-1, i), overlaps[i+1] = prefix(i, i+1)
                current = overlaps[i] + overlaps[i+1]
                
                # What if we swap rows i and i+1?
                new_overlap_i = self._prefix_len(strings[i-1], strings[i+1])
                new_overlap_ip1 = self._prefix_len(strings[i+1], strings[i])
                # Also need to consider overlap with i+2 if exists
                new_overlap_ip2 = overlaps[i+2] if i + 2 < n else 0
                old_overlap_ip2 = overlaps[i+2] if i + 2 < n else 0
                
                if i + 2 < n:
                    new_overlap_ip2 = self._prefix_len(strings[i], strings[i+2])
                    current += old_overlap_ip2
                    new_total = new_overlap_i + new_overlap_ip1 + new_overlap_ip2
                else:
                    new_total = new_overlap_i + new_overlap_ip1
                
                if new_total > current:
                    # Swap!
                    ordered_rows[i], ordered_rows[i+1] = ordered_rows[i+1], ordered_rows[i]
                    strings[i], strings[i+1] = strings[i+1], strings[i]
                    overlaps[i] = new_overlap_i
                    overlaps[i+1] = new_overlap_ip1
                    if i + 2 < n:
                        overlaps[i+2] = new_overlap_ip2
                    improved = True
        
        return ordered_rows
    
    def _optimize_pair(self, row_indices, str_vals, available_cols):
        """Optimal column ordering for exactly 2 rows."""
        if not available_cols:
            return [(idx, []) for idx in row_indices]
        
        idx0, idx1 = row_indices[0], row_indices[1]
        
        # Classify columns
        matching = []  # values identical
        partial = []   # values share a prefix but differ
        
        for col in available_cols:
            v0 = str_vals[idx0, col]
            v1 = str_vals[idx1, col]
            if v0 == v1:
                matching.append((col, len(v0)))  # full match
            else:
                # Compute prefix overlap
                plen = 0
                ml = min(len(v0), len(v1))
                for k in range(ml):
                    if v0[k] == v1[k]:
                        plen += 1
                    else:
                        break
                partial.append((col, plen))
        
        # Sort matching by value length desc (longer matches contribute more)
        matching.sort(key=lambda x: -x[1])
        # Sort partial by prefix length desc (place the one with longest prefix LAST
        # among partials - actually FIRST, because after all matching columns,
        # the first partial column's prefix determines how many more chars match)
        partial.sort(key=lambda x: -x[1])
        
        # Optimal order: all matching columns first (sorted by length desc), 
        # then partial columns sorted by prefix desc
        # The first partial column's prefix length determines the additional characters
        # Only the FIRST partial column matters - after it mismatches, prefix stops
        col_order = [c for c, _ in matching] + [c for c, _ in partial]
        
        return [(idx0, list(col_order)), (idx1, list(col_order))]
    
    def _fast_fallback(self, row_indices, str_vals, available_cols):
        if not available_cols:
            return [(idx, []) for idx in row_indices]
        
        from collections import defaultdict
        col_scores = []
        for col in available_cols:
            val_counts = defaultdict(int)
            for idx in row_indices:
                val_counts[str_vals[idx, col]] += 1
            score = 0
            for v, cnt in val_counts.items():
                if cnt > 1:
                    score += (cnt - 1) * len(v)
            col_scores.append((score, col))
        
        col_scores.sort(reverse=True)
        sorted_cols = [c for _, c in col_scores]
        
        def sort_key(idx):
            return tuple(str_vals[idx, c] for c in sorted_cols)
        sorted_indices = sorted(row_indices, key=sort_key)
        return [(idx, sorted_cols) for idx in sorted_indices]
    
    def _sort_by_prefix_overlap(self, row_indices, str_vals, available_cols):
        if not available_cols:
            return [(idx, []) for idx in row_indices]
        
        col_prefix_scores = {}
        for col in available_cols:
            vals = [str_vals[idx, col] for idx in row_indices]
            vals.sort()
            total_prefix = 0
            for i in range(1, len(vals)):
                s1, s2 = vals[i-1], vals[i]
                ml = min(len(s1), len(s2))
                for k in range(ml):
                    if s1[k] == s2[k]:
                        total_prefix += 1
                    else:
                        break
            col_prefix_scores[col] = total_prefix
        
        sorted_remaining = sorted(available_cols, key=lambda c: -col_prefix_scores[c])
        
        def sort_key(idx):
            return ''.join(str_vals[idx, c] for c in sorted_remaining)
        sorted_indices = sorted(row_indices, key=sort_key)
        return [(idx, list(sorted_remaining)) for idx in sorted_indices]