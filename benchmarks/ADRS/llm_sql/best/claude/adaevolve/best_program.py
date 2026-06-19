# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from functools import lru_cache
from collections import Counter
import networkx as nx
import numpy as np


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

    def compute_value_frequencies(self, df: pd.DataFrame, all_cols: List[str]) -> Dict:
        """Precompute frequency of each value across all columns."""
        freq_dict = {}
        for col in all_cols:
            value_counts = df[col].value_counts()
            for val, count in value_counts.items():
                if val not in freq_dict:
                    freq_dict[val] = 0
                freq_dict[val] += count
        return freq_dict
    
    def compute_column_ordering_dp(self, curr_row: pd.Series, prev_row: pd.Series, 
                                   all_cols: List[str], value_freq: Dict, 
                                   remaining_rows: int) -> List[str]:
        """
        Dynamic programming approach: compute optimal column ordering for current row
        based on prefix continuation with previous row and value frequencies.
        
        Score = (match_with_prev) * len(value)^2 * (remaining_frequency / remaining_rows)
        """
        if prev_row is None:
            # First row: order by value length * frequency
            scores = []
            for col in all_cols:
                val = curr_row[col]
                val_len = self.val_len.get(val, 0)
                freq = value_freq.get(val, 1)
                score = val_len * freq
                scores.append((col, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            return [col for col, _ in scores]
        
        # Find columns that match previous row
        matching_cols = []
        non_matching_cols = []
        
        for col in all_cols:
            if curr_row[col] == prev_row[col]:
                matching_cols.append(col)
            else:
                non_matching_cols.append(col)
        
        # Score matching columns by value length (to maximize prefix hit score)
        match_scores = []
        for col in matching_cols:
            val = curr_row[col]
            val_len = self.val_len.get(val, 0)
            match_scores.append((col, val_len))
        match_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Score non-matching columns by length * frequency
        non_match_scores = []
        for col in non_matching_cols:
            val = curr_row[col]
            val_len = self.val_len.get(val, 0)
            freq = value_freq.get(val, 1)
            score = val_len * (freq / max(1, remaining_rows))
            non_match_scores.append((col, score))
        non_match_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Combine: matching columns first (ordered by length), then non-matching
        ordered = [col for col, _ in match_scores] + [col for col, _ in non_match_scores]
        return ordered

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

    def dp_prefix_reorder(self, df: pd.DataFrame, all_cols: List[str]) -> pd.DataFrame:
        """
        Dynamic programming approach with prefix-aware column ordering.
        For each row, compute optimal column ordering based on previous row's values
        to maximize prefix continuation and overall hit score.
        Uses vectorization where possible for efficiency.
        """
        if df.empty:
            return df
        
        # Precompute value frequencies for scoring
        value_freq = self.compute_value_frequencies(df, all_cols)
        
        # Convert to numpy for faster access
        df_values = df[all_cols].values
        n_rows, n_cols = df_values.shape
        
        # Pre-sort rows by first few columns to group similar rows
        sort_cols = sorted(all_cols, 
                          key=lambda c: sum(self.val_len.get(v, 0) * value_freq.get(v, 1) 
                                          for v in df[c].unique()), 
                          reverse=True)[:min(3, len(all_cols))]
        df_sorted = df.sort_values(by=sort_cols, kind='stable')
        
        result_rows = []
        prev_row = None
        final_cols = None
        
        for idx in range(len(df_sorted)):
            curr_row = df_sorted.iloc[idx]
            remaining_rows = len(df_sorted) - idx
            
            # Compute optimal column ordering using DP approach
            if prev_row is None:
                ordering = self.compute_column_ordering_dp(
                    curr_row, None, all_cols, value_freq, remaining_rows
                )
            else:
                ordering = self.compute_column_ordering_dp(
                    curr_row, prev_row, all_cols, value_freq, remaining_rows
                )
            
            # Build reordered row
            reordered_values = [curr_row[col] for col in ordering]
            if 'original_index' in df_sorted.columns:
                reordered_values.append(curr_row['original_index'])
                final_cols = ordering + ['original_index']
            else:
                final_cols = ordering
            
            result_rows.append(reordered_values)
            
            # Update prev_row with reordered values for next iteration
            prev_row = pd.Series([curr_row[col] for col in ordering], index=ordering)
        
        result_df = pd.DataFrame(result_rows, columns=final_cols)
        return result_df

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
        initial_value_counts = Counter(recurse_df.stack())
        self.val_len = {val: self.calculate_length(val) for val in initial_value_counts.keys()}

        self.row_stop = row_stop if row_stop else len(recurse_df)
        self.col_stop = col_stop if col_stop else len(recurse_df.columns.tolist())
        print("*" * 80)
        print(f"DF columns = {df.columns}")
        # print(f"Early stop = {early_stop}")
        # print(f"Row recursion stop depth = {self.row_stop}, Column recursion stop depth = {self.col_stop}")
        print("*" * 80)

        # Apply dynamic programming prefix-aware reordering
        self.num_cols = len(recurse_df.columns)
        all_cols = [c for c in recurse_df.columns if c != 'original_index']
        reordered_df = self.dp_prefix_reorder(recurse_df, all_cols)

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

        # sort by the first column to get the final order
        final_df = final_df.sort_values(by=final_df.columns.to_list(), axis=0)
        return final_df, []

# EVOLVE-BLOCK-END