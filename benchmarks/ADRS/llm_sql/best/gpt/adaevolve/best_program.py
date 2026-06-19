# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter
import numpy as np
import networkx as nx


class Evolved(Algorithm):
    """
    GGR algorithm
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

        self.dep_graph = None  # NOTE: not used, for one way dependency

        self.num_rows = 0
        self.num_cols = 0
        self.column_stats = None
        self.val_len = None
        self.row_stop = None
        self.col_stop = None
        self.base = 2000

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0) -> str:
        # NOTE: recalculate value counts and length for each value
        value_counts = Counter(df.stack())
        weighted_counts = {val: self.val_len.get(val, 0) * (count - 1) for val, count in value_counts.items() if pd.notna(val)}  # if count > 1
        if not weighted_counts:
            return None
        max_group_val, max_weighted_count = max(weighted_counts.items(), key=lambda x: x[1])
        if max_weighted_count < early_stop:
            return None
        return max_group_val

    def reorder_columns_for_value(self, row, value, column_names, grouped_rows_len: int = 1):
        # cols_with_value will now use attribute access instead of indexing with row[]
        cols_with_value = []
        for idx, col in enumerate(column_names):
            if hasattr(row, col) and getattr(row, col) == value:
                cols_with_value.append(col)
            elif hasattr(row, col.replace(" ", "_")) and getattr(row, col.replace(" ", "_")) == value:
                cols_with_value.append(col)
            else:
                attr_name = f"_{idx}"
                if hasattr(row, attr_name) and getattr(row, attr_name) == value:
                    cols_with_value.append(attr_name)

        if self.dep_graph is not None and grouped_rows_len > 1:
            # NOTE: experimental
            reordered_cols = []
            for col in cols_with_value:
                dependent_cols = self.get_dependent_columns(col)

                # check if dependent columns are in row, and if column exists in row attributes
                valid_dependent_cols = []
                for idx, dep_col in enumerate(dependent_cols):
                    if hasattr(row, dep_col):
                        valid_dependent_cols.append(dep_col)
                    elif hasattr(row, dep_col.replace(" ", "_")):
                        valid_dependent_cols.append(dep_col)
                    else:
                        attr_name = f"_{idx}"
                        if hasattr(row, attr_name):
                            valid_dependent_cols.append(dep_col)

                reordered_cols.extend([col] + valid_dependent_cols)
            cols_without_value = [col for col in column_names if col not in reordered_cols]
            reordered_cols.extend(cols_without_value)
            assert len(reordered_cols) == len(
                column_names
            ), f"Reordered cols len: {len(reordered_cols)}  Original cols len: {len(column_names)}"
            return [getattr(row, col) for col in reordered_cols], cols_with_value
        else:
            cols_without_value = []
            for idx, col in enumerate(column_names):
                if hasattr(row, col) and getattr(row, col) != value:
                    cols_without_value.append(col)
                elif hasattr(row, col.replace(" ", "_")) and getattr(row, col.replace(" ", "_")) != value:
                    cols_without_value.append(col)
                else:
                    # Handle some edge cases
                    attr_name = f"_{idx}"
                    if hasattr(row, attr_name) and getattr(row, attr_name) != value:
                        cols_without_value.append(attr_name)

            reordered_cols = cols_with_value + cols_without_value
            assert len(reordered_cols) == len(
                column_names
            ), f"Reordered cols len: {len(reordered_cols)}  Original cols len: {len(column_names)}"
            return [getattr(row, col) for col in reordered_cols], cols_with_value

    def get_dependent_columns(self, col: str) -> List[str]:
        if self.dep_graph is None or not self.dep_graph.has_node(col):
            return []
        return list(nx.descendants(self.dep_graph, col))

    @lru_cache(maxsize=None)
    def get_cached_dependent_columns(self, col: str) -> List[str]:
        return self.get_dependent_columns(col)

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        num_rows, column_stats = self.calculate_col_stats(df, enable_index=True)
        reordered_columns = [col for col, _, _, _ in column_stats]
        reordered_df = df[reordered_columns]

        assert reordered_df.shape == df.shape
        column_orderings = [reordered_columns] * num_rows

        if row_sort:
            reordered_df = reordered_df.sort_values(by=reordered_columns, axis=0)

        return reordered_df, column_orderings

    def column_recursion(self, result_df, max_value, grouped_rows, row_stop, col_stop, early_stop):
        cols_settled = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.reorder_columns_for_value, row, max_value, grouped_rows.columns.tolist(), len(grouped_rows))
                for row in grouped_rows.itertuples(index=False)
            ]
            for i, future in enumerate(as_completed(futures)):
                reordered_row, cols_settled = future.result()
                result_df.loc[i] = reordered_row

        grouped_value_counts = Counter()

        if not result_df.empty:
            result_df = result_df.astype(object)
            # Group by the first column
            grouped_result_df = result_df.groupby(result_df.columns[0])
            grouped_value_counts = Counter(grouped_rows.stack())  # this is still faster than updating from cached value counts

            for _, group in grouped_result_df:
                if group[group.columns[0]].iloc[0] != max_value:
                    continue

                dependent_cols = self.get_cached_dependent_columns(group.columns[0])
                length_of_settle_cols = len(cols_settled)

                if dependent_cols:
                    assert length_of_settle_cols >= 1, f"Dependent columns should be no less than 1, but got {length_of_settle_cols}"

                    # test the first length_of_settle_cols columns, each column has nunique == 1
                    for col in group.columns[:length_of_settle_cols]:
                        assert group[col].nunique() == 1, f"Column {col} should have nunique == 1, but got {group[col].nunique()}"

                    # drop all the settled columns and reorder the rest
                    group_remainder = group.iloc[:, length_of_settle_cols:]
                else:
                    group_remainder = group.iloc[:, 1:]

                grouped_remainder_value_counts = Counter(group_remainder.stack())

                reordered_group_remainder, _ = self.recursive_reorder(
                    group_remainder, grouped_remainder_value_counts, early_stop=early_stop, row_stop=row_stop, col_stop=col_stop + 1
                )
                # Update the group with the reordered columns
                if dependent_cols:
                    group.iloc[:, length_of_settle_cols:] = reordered_group_remainder.values
                else:
                    group.iloc[:, 1:] = reordered_group_remainder.values

                result_df.update(group)
                break

        return result_df, grouped_value_counts

    def recursive_reorder(
        self,
        df: pd.DataFrame,
        value_counts: Dict,
        early_stop: int = 0,
        original_columns: List[str] = None,
        row_stop: int = 0,
        col_stop: int = 0,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        if df.empty or len(df.columns) == 0 or len(df) == 0:
            return df, []

        if self.row_stop is not None and row_stop >= self.row_stop:
            return self.fixed_reorder(df)

        if self.col_stop is not None and col_stop >= self.col_stop:
            return self.fixed_reorder(df)

        if original_columns is None:
            original_columns = df.columns.tolist()

        # Find the max group value using updated counts
        max_value = self.find_max_group_value(df, value_counts, early_stop=early_stop)
        if max_value is None:
            # If there is no max value, then fall back to fixed reorder
            return self.fixed_reorder(df)

        grouped_rows = df[df.isin([max_value]).any(axis=1)]
        remaining_rows = df[~df.isin([max_value]).any(axis=1)]

        # If there is no grouped rows, return the original DataFrame
        if grouped_rows.empty:
            return self.fixed_reorder(df)

        result_df = pd.DataFrame(columns=df.columns)

        reordered_remaining_rows = pd.DataFrame(columns=df.columns)  # Initialize empty dataframe first

        # Column Recursion
        result_df, grouped_value_counts = self.column_recursion(result_df, max_value, grouped_rows, row_stop, col_stop, early_stop)

        remaining_value_counts = value_counts - grouped_value_counts  # Approach 1 - update remaining value counts with subtraction

        # Row Recursion
        reordered_remaining_rows, _ = self.recursive_reorder(
            remaining_rows, remaining_value_counts, early_stop=early_stop, row_stop=row_stop + 1, col_stop=col_stop
        )
        old_column_names = result_df.columns.tolist()
        result_cols_reset = result_df.reset_index(drop=True)
        result_rows_reset = reordered_remaining_rows.reset_index(drop=True)
        final_result_df = pd.DataFrame(result_cols_reset.values.tolist() + result_rows_reset.values.tolist())

        if row_stop == 0 and col_stop == 0:
            final_result_df.columns = old_column_names
            final_result_df.columns = final_result_df.columns.tolist()[:-1] + ["original_index"]

        return final_result_df, []

    def recursive_split_and_reorder(self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0):
        """
        Recursively split the DataFrame into halves until the size is <= 1000, then apply the recursive reorder function.
        """
        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack())
            return self.recursive_reorder(df, initial_value_counts, early_stop, original_columns, row_stop=0, col_stop=0)[0]

        mid_index = len(df) // 2
        df_top_half = df.iloc[:mid_index]
        df_bottom_half = df.iloc[mid_index:]

        with ThreadPoolExecutor() as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, df_top_half, original_columns, early_stop)
            future_bottom = executor.submit(self.recursive_split_and_reorder, df_bottom_half, original_columns, early_stop)

        reordered_top_half = future_top.result()
        reordered_bottom_half = future_bottom.result()

        assert reordered_bottom_half.shape == df_bottom_half.shape
        reordered_df = pd.concat([reordered_top_half, reordered_bottom_half], axis=0, ignore_index=True)

        assert reordered_df.shape == df.shape

        return reordered_df

    @lru_cache(maxsize=None)
    def calculate_length(self, value):
        if isinstance(value, bool):
            return 4**2
        if isinstance(value, (int, float)):
            return len(str(value)) ** 2
        if isinstance(value, str):
            return len(value) ** 2
        return 0

    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        """
        Spectral row seriation (Fiedler vector) + prefix-aware greedy columns with lookahead.
        - Rank columns by reuse mass and avg len^2 to build a base order.
        - Build a sparse similarity graph within local bins of top-K columns and weight edges by equal-value squared-length mass.
        - Order rows per connected component by the Fiedler vector; fallback to stable sort if SciPy unavailable.
        - Refine base column order using adjacency equality mass on the ordered rows.
        - Per-row: keep longest equal prefix vs previous, then next-equal columns by length^2 and equality rate, then remainder.
        """
        initial_df = df.copy()

        nrows, ncols = df.shape
        if nrows == 0 or ncols == 0:
            base_cols = list(df.columns)
            return df, [base_cols for _ in range(nrows)]

        # String-cast for exact match semantics
        df_str = df.astype(str)

        # Choose clustering-friendly columns (avoid near-unique)
        nunique = df_str.nunique(dropna=False)
        threshold = int(len(df_str) * distinct_value_threshold)
        cluster_cols = [c for c in df_str.columns if nunique[c] <= threshold] or list(df_str.columns)

        # Score columns by reuse mass and avg len^2 to create a base order
        avg_len2: Dict[str, float] = {c: float((df_str[c].str.len() ** 2).mean() or 0.0) for c in df_str.columns}
        pair_weight: Dict[str, float] = {}
        for c in df_str.columns:
            vc = df_str[c].value_counts(dropna=False)
            w = 0.0
            for v, cnt in vc.items():
                if cnt <= 1 or v == "nan":
                    continue
                w += (cnt - 1) * (len(v) ** 2)
            pair_weight[c] = w
        base_order = sorted(df_str.columns, key=lambda c: (-pair_weight[c], -avg_len2[c], c))

        # Row sequencing via spectral seriation (Fiedler vector), with robust fallback
        df_str_sorted = df_str
        used_spectral = False
        try:
            from scipy.sparse import csr_matrix  # type: ignore
            from scipy.sparse import diags  # type: ignore
            from scipy.sparse import csgraph  # type: ignore
            from scipy.sparse.linalg import eigsh  # type: ignore

            if nrows > 1:
                K = min(6, len(base_order))
                feat_cols = [c for c in base_order if c in cluster_cols][:K] or base_order[:K]
                if K > 0:
                    arr_feat = df_str[feat_cols].to_numpy(dtype=object)
                    len2_feat = df_str[feat_cols].apply(lambda s: s.str.len() ** 2).to_numpy(dtype=np.int64)

                    # Bin by first B features to localize neighbor search
                    B = min(2, K)
                    keys = [tuple(arr_feat[i, :B]) if B > 0 else tuple() for i in range(nrows)]
                    bins: Dict[tuple, List[int]] = {}
                    for i, k in enumerate(keys):
                        bins.setdefault(k, []).append(i)

                    rows_e: List[int] = []
                    cols_e: List[int] = []
                    data_e: List[float] = []
                    window = 3

                    # Connect small-window neighbors within each bin; edge weights by equal-value length^2 mass
                    for idxs in bins.values():
                        m = len(idxs)
                        if m <= 1:
                            continue
                        idxs_sorted = sorted(idxs, key=lambda i: tuple(arr_feat[i, :K]))
                        for p in range(m):
                            i_pos = idxs_sorted[p]
                            for t in range(1, window + 1):
                                q = p + t
                                if q >= m:
                                    break
                                j_pos = idxs_sorted[q]
                                eq_vec = (arr_feat[i_pos, :] == arr_feat[j_pos, :])
                                if hasattr(eq_vec, "astype"):
                                    eq_vec = eq_vec.astype(bool)
                                w_ij = float((eq_vec * len2_feat[i_pos, :]).sum())
                                if w_ij <= 0.0:
                                    continue
                                rows_e.append(i_pos); cols_e.append(j_pos); data_e.append(w_ij)
                                rows_e.append(j_pos); cols_e.append(i_pos); data_e.append(w_ij)

                    if data_e:
                        W = csr_matrix((data_e, (rows_e, cols_e)), shape=(nrows, nrows))
                        W = W.maximum(W.T)

                        # Connected components for scalable spectral ordering
                        ncomp, labels = csgraph.connected_components(W, directed=False)
                        comp_ids = list(range(ncomp))
                        comp_order = sorted(comp_ids, key=lambda cid: int(np.min(np.where(labels == cid)[0])) if np.any(labels == cid) else cid)

                        perm_all: List[int] = []
                        for cid in comp_order:
                            idxs = np.where(labels == cid)[0]
                            m = len(idxs)
                            if m <= 1:
                                perm_all.extend(idxs.tolist())
                                continue
                            W_sub = W[idxs][:, idxs]
                            d = np.array(W_sub.sum(axis=1)).ravel()
                            L_sub = diags(d) - W_sub
                            try:
                                vals, vecs = eigsh(L_sub, k=2, which="SM", tol=1e-3)
                                fied = vecs[:, 1]
                                order = np.argsort(fied, kind="mergesort")
                                perm_all.extend(idxs[order].tolist())
                            except Exception:
                                order = sorted(range(m), key=lambda r: tuple(arr_feat[idxs[r], :K]))
                                perm_all.extend([idxs[r] for r in order])

                        df_str_sorted = df_str.iloc[perm_all]
                        used_spectral = True
        except Exception:
            used_spectral = False

        if not used_spectral:
            # Fallback stable sort by top-K columns from base order
            topk = min(8, len(base_order))
            sort_cols = [c for c in base_order if c in cluster_cols][:topk] or base_order[:topk]
            try:
                df_str_sorted = df_str.sort_values(by=sort_cols, kind="mergesort")
            except Exception:
                df_str_sorted = df_str

        # Refine base order using adjacency equality stats on the RCM-ordered rows
        arr_tmp = df_str_sorted[base_order].to_numpy(dtype=object)
        if nrows > 1:
            eq_adj = (arr_tmp[1:, :] == arr_tmp[:-1, :])
            eq_counts = eq_adj.sum(axis=0)
            total_pairs = nrows - 1
        else:
            eq_adj = np.zeros((0, len(base_order)), dtype=bool)
            eq_counts = np.zeros((len(base_order),), dtype=int)
            total_pairs = 1

        col_eq_rate: Dict[str, float] = {}
        col_equal_mass: Dict[str, float] = {}
        lengths2_cols_tmp = [(df_str_sorted[c].str.len().to_numpy(dtype=np.int32) ** 2) for c in base_order]
        lengths2_tmp = np.column_stack(lengths2_cols_tmp) if lengths2_cols_tmp else np.zeros((nrows, 0), dtype=np.int32)
        for j, c in enumerate(base_order):
            cnt = int(eq_counts[j])
            col_eq_rate[c] = cnt / max(total_pairs, 1)
            if cnt > 0 and nrows > 1:
                col_equal_mass[c] = float((lengths2_tmp[1:, j] * eq_adj[:, j]).sum())
            else:
                col_equal_mass[c] = 0.0
        base_order = sorted(base_order, key=lambda c: (-col_eq_rate[c], -col_equal_mass[c], -avg_len2[c], c))

        # Build arrays on RCM-ordered rows and refined base columns
        arr = df_str_sorted[base_order].to_numpy(dtype=object)
        lengths2_cols2 = [(df_str_sorted[c].str.len().to_numpy(dtype=np.int32) ** 2) for c in base_order]
        lengths2 = np.column_stack(lengths2_cols2) if lengths2_cols2 else np.zeros((nrows, 0), dtype=np.int32)
        if nrows > 1:
            eq_next = (arr[:-1, :] == arr[1:, :])
        else:
            eq_next = np.zeros((0, len(base_order)), dtype=bool)

        idx_in_base = {c: i for i, c in enumerate(base_order)}

        # Construct per-row column orderings: prefix vs previous, then next-equal by len^2, then remainder
        orderings: List[List[str]] = []
        prev_order = list(base_order)

        for i in range(nrows):
            if i == 0:
                prefix_idx: List[int] = []
            else:
                prefix_idx = []
                for c in prev_order:
                    j = idx_in_base[c]
                    if arr[i, j] == arr[i - 1, j]:
                        prefix_idx.append(j)
                    else:
                        break

            used_idx = set(prefix_idx)

            lookahead_idx: List[int] = []
            if i < nrows - 1:
                eq_cols = np.where(eq_next[i])[0].tolist()
                eq_cols = [j for j in eq_cols if j not in used_idx]
                if eq_cols:
                    eq_cols.sort(key=lambda j: (-int(lengths2[i, j]), -col_eq_rate[base_order[j]], j))
                    lookahead_idx = eq_cols

            used_tail = used_idx | set(lookahead_idx)
            remainder_idx = [j for j in range(len(base_order)) if j not in used_tail]

            order_idx = (prefix_idx + lookahead_idx + remainder_idx) if i > 0 else (lookahead_idx + remainder_idx)
            order_i = [base_order[j] for j in order_idx]
            orderings.append(order_i)
            prev_order = order_i

        final_df = initial_df.loc[df_str_sorted.index, base_order]
        return final_df, orderings

# EVOLVE-BLOCK-END