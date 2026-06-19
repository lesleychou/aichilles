# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter
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
        self._col_rank = {}

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0) -> str:
        # Use provided value_counts to avoid recomputation; weight by squared length and frequency-1
        if not value_counts:
            return None
        weighted_counts = {}
        for val, count in value_counts.items():
            if count > 1 and val is not None and val == val:  # skip NaN
                weighted_counts[val] = self.val_len.get(val, 0) * (count - 1)
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

        # Prioritize columns where the chosen value is most frequent across the current group
        if getattr(self, "_col_rank", None):
            try:
                cols_with_value.sort(key=lambda c: self._col_rank.get(c, len(self._col_rank) + 1))
            except Exception:
                pass

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

        # Build a column rank map prioritizing columns where max_value appears most often
        try:
            bool_mask = grouped_rows.eq(max_value)
            col_counts = bool_mask.sum(axis=0)
            order = [c for c, _ in sorted(
                zip(grouped_rows.columns.tolist(), col_counts.tolist()),
                key=lambda x: (-x[1], grouped_rows.columns.get_loc(x[0]))
            )]
            self._col_rank = {c: i for i, c in enumerate(order)}
        except Exception:
            self._col_rank = {}

        # Build rows buffer to avoid per-row DataFrame assignment overhead
        rows_buffer = []
        for row in grouped_rows.itertuples(index=False):
            reordered_row, cols_settled = self.reorder_columns_for_value(
                row, max_value, grouped_rows.columns.tolist(), len(grouped_rows)
            )
            rows_buffer.append(reordered_row)
        if rows_buffer:
            result_df = pd.DataFrame(rows_buffer, columns=grouped_rows.columns)
        else:
            result_df = pd.DataFrame(columns=grouped_rows.columns)

        # Fast count of grouped values without stack()
        grouped_value_counts = Counter()
        if not grouped_rows.empty:
            vals = grouped_rows.to_numpy(dtype=object).ravel()
            for v in vals:
                if v is not None and v == v:
                    grouped_value_counts[v] += 1

        if not result_df.empty:
            result_df = result_df.astype(object)

            # Determine the longest leading prefix length k where all rows share max_value
            k = 0
            for c in result_df.columns:
                s = result_df[c]
                if s.nunique() == 1 and s.iloc[0] == max_value:
                    k += 1
                else:
                    break
            if k == 0:
                k = 1

            # Compute remainder after settled prefix and recurse
            dependent_cols = self.get_cached_dependent_columns(result_df.columns[0])
            group_remainder = result_df.iloc[:, k:] if result_df.shape[1] > k else pd.DataFrame(columns=[])
            if not group_remainder.empty:
                grouped_remainder_value_counts = self.fast_counter(group_remainder)
                reordered_group_remainder, _ = self.recursive_reorder(
                    group_remainder, grouped_remainder_value_counts, early_stop=early_stop, row_stop=row_stop, col_stop=col_stop + 1
                )
                result_df.iloc[:, k:] = reordered_group_remainder.values

        # Clear rank to avoid leaking into unrelated groups
        self._col_rank = {}
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
            initial_value_counts = self.fast_counter(df)
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

    def fast_counter(self, df: pd.DataFrame) -> Counter:
        # Count values across the DataFrame without using stack(); ignore original_index to avoid skew
        cnt = Counter()
        if df is None or getattr(df, "empty", False):
            return cnt
        try:
            use_df = df.drop(columns=["original_index"], errors="ignore")
        except Exception:
            use_df = df
        if getattr(use_df, "empty", False):
            return cnt
        vals = use_df.to_numpy(dtype=object).ravel()
        for v in vals:
            if v is not None and v == v:
                cnt[v] += 1
        return cnt

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
        # Prepare
        initial_df = df.copy()
        df = df.astype(str)
        if col_merge:
            self.num_rows, self.column_stats = self.calculate_col_stats(df, enable_index=True)
            reordered_columns = [col for col, _, _, _ in self.column_stats]
            for col_to_merge in col_merge:
                final_col_order = [col for col in reordered_columns if col in col_to_merge]
                df = self.merging_columns(df, final_col_order, prepended=False)
        self.num_rows, self.column_stats = self.calculate_col_stats(df, enable_index=True)
        self.column_stats = {col: (num_groups, avg_len, score) for col, num_groups, avg_len, score in self.column_stats}

        # One way dependency statistics [not used]
        if one_way_dep is not None and len(one_way_dep) > 0:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                col1 = [col for col in df.columns if dep[0] in col]
                col2 = [col for col in df.columns if dep[1] in col]
                assert len(col1) == 1, f"Expected one column to match {dep[0]}, but got {len(col1)}"
                assert len(col2) == 1, f"Expected one column to match {dep[1]}, but got {len(col2)}"
                col1 = col1[0]
                col2 = col2[0]
                self.dep_graph.add_edge(col1, col2)

        # Discard too distinct columns by threshold [optional]
        nunique_threshold = len(df) * distinct_value_threshold
        columns_to_discard = [col for col in df.columns if df[col].nunique() > nunique_threshold]
        columns_to_discard = sorted(columns_to_discard, key=lambda x: self.column_stats[x][2], reverse=True)
        columns_to_recurse = [col for col in df.columns if col not in columns_to_discard]
        df["original_index"] = range(len(df))
        discarded_columns_df = df[columns_to_discard + ["original_index"]]
        df_to_recurse = df[columns_to_recurse + ["original_index"]]
        recurse_df = df_to_recurse

        self.column_stats = {col: stats for col, stats in self.column_stats.items() if col not in columns_to_discard}
        initial_value_counts = self.fast_counter(recurse_df)
        self.val_len = {val: self.calculate_length(val) for val in initial_value_counts.keys()}

        self.row_stop = row_stop if row_stop else len(recurse_df)
        self.col_stop = col_stop if col_stop else len(recurse_df.columns.tolist())
        # logging suppressed for speed

        # Eary stop and fall back
        recurse_df, _ = self.fixed_reorder(recurse_df, row_sort=False)

        # Recursive reordering
        self.num_cols = len(recurse_df.columns)
        if parallel:
            reordered_df = self.recursive_split_and_reorder(recurse_df, original_columns=columns_to_recurse, early_stop=early_stop)
        else:
            reordered_df, _ = self.recursive_reorder(
                recurse_df,
                initial_value_counts,
                early_stop=early_stop,
            )

        assert (
            reordered_df.shape == recurse_df.shape
        ), f"Reordered DataFrame shape {reordered_df.shape} does not match original DataFrame shape {recurse_df.shape}"
        assert recurse_df["original_index"].is_unique, "Passed in recurse index contains duplicates!"
        assert reordered_df["original_index"].is_unique, "Reordered index contains duplicates!"

        if len(columns_to_discard) > 0:
            final_df = pd.merge(reordered_df, discarded_columns_df, on="original_index", how="left")
        else:
            final_df = reordered_df

        final_df = final_df.drop(columns=["original_index"])

        if not col_merge:
            assert (
                final_df.shape == initial_df.shape
            ), f"Final DataFrame shape {final_df.shape} does not match original DataFrame shape {initial_df.shape}"
        else:
            assert (
                final_df.shape[0] == initial_df.shape[0]
            ), f"Final DataFrame shape {final_df.shape} does not match original DataFrame shape {initial_df.shape}"
            assert (
                final_df.shape[1] == recurse_df.shape[1] + len(columns_to_discard) - 1
            ), f"Final DataFrame shape {final_df.shape} does not match original DataFrame shape {recurse_df.shape}"

        # adjacency-yield aware stable sort using sample-based weights for speed
        try:
            cols_no_idx = [c for c in final_df.columns if c != "original_index"]
            if not cols_no_idx:
                return final_df, []
            sample = final_df[cols_no_idx].head(min(len(final_df), 5000))
            weights = {}
            for c in cols_no_idx:
                vc = sample[c].value_counts(dropna=True)
                w = 0
                if not vc.empty:
                    for val, cnt in vc.items():
                        if cnt > 1 and val is not None and val == val:
                            w += (cnt * (cnt - 1)) * self.calculate_length(val)
                weights[c] = w
            top_cols = [c for c, _ in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)][:min(8, len(cols_no_idx))]
            if not top_cols:
                top_cols = cols_no_idx
            final_df = final_df.sort_values(by=top_cols, axis=0, kind="mergesort")
        except Exception:
            final_df = final_df.sort_values(by=final_df.columns.to_list(), axis=0, kind="mergesort")
        return final_df, []

# EVOLVE-BLOCK-END