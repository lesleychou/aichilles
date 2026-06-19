import pandas as pd
from solver import Algorithm


class Evolved(Algorithm):
    """
    Multi-phase algorithm with optimized exponent (1.72 for narrow, 2.37 for wide).
    """

    def reorder(self, df: pd.DataFrame, **kwargs):
        if df.empty:
            return df

        # Order columns by uniqueness
        cols_ordered = list(sorted(df.columns, key=lambda c: df[c].nunique()))
        
        num_cols = len(df.columns)
        
        if num_cols <= 8:
            length_exp = 1.72
        else:
            length_exp = 2.37
        
        if num_cols <= 8:
            # Phase 1: Pre-permute all rows by value length
            df_permuted_rows = []
            for idx, row in df[cols_ordered].iterrows():
                row_values = [str(v) for v in row.values]
                perm = sorted(row_values, key=lambda v: (-len(v), v))
                df_permuted_rows.append(perm)
            
            df_permuted = pd.DataFrame(df_permuted_rows, columns=cols_ordered)

            # Phase 2: Sort rows lexicographically on permuted data
            if num_cols <= 3:
                sort_cols = cols_ordered[:1]
            elif num_cols <= 4:
                sort_cols = cols_ordered[:2]
            else:
                sort_cols = cols_ordered[:3]
            sorted_df = df_permuted.sort_values(
                by=sort_cols,
                kind="mergesort"
            ).reset_index(drop=True)

            result = self._apply_permutation_narrow(sorted_df, cols_ordered, length_exp)
        else:
            df_ordered = df[cols_ordered]
            sort_cols = cols_ordered[:3]
            sorted_df = df_ordered.sort_values(
                by=sort_cols,
                kind="mergesort"
            ).reset_index(drop=True)
            
            result = self._apply_permutation_wide(sorted_df, cols_ordered, length_exp)
        
        return result

    def _apply_permutation_narrow(self, df_ordered, cols_ordered, length_exp):
        result_rows = []
        
        value_freq = {}
        for idx, row in df_ordered.iterrows():
            for v in row.values:
                v_str = str(v)
                value_freq[v_str] = value_freq.get(v_str, 0) + 1
        
        prev_values = None
        prev_str = ""
        
        for row_idx, row in df_ordered.iterrows():
            row_values = [str(v) for v in row.values]
            
            if prev_values is not None:
                perm1 = self._smart_perm_freq(row_values, prev_values, value_freq, length_exp)
                
                str1_start = str(perm1[0])
                if not prev_str.startswith(str1_start):
                    perm2 = self._freq_perm(row_values, value_freq, length_exp)
                    str1_full = "".join(perm1)
                    str2_full = "".join(perm2)
                    
                    match1 = 0
                    for i in range(min(len(prev_str), len(str1_full))):
                        if prev_str[i] == str1_full[i]:
                            match1 += 1
                        else:
                            break
                    
                    match2 = 0
                    for i in range(min(len(prev_str), len(str2_full))):
                        if prev_str[i] == str2_full[i]:
                            match2 += 1
                        else:
                            break
                    
                    if match2 > match1:
                        perm = perm2
                        prev_str = str2_full
                    else:
                        perm = perm1
                        prev_str = str1_full
                else:
                    perm = perm1
                    prev_str = "".join(perm1)
            else:
                perm = row_values
                prev_str = "".join(perm)
            
            result_rows.append(perm)
            prev_values = perm
        
        return pd.DataFrame(result_rows, columns=cols_ordered)
    
    def _apply_permutation_wide(self, df_ordered, cols_ordered, length_exp):
        result_rows = []
        
        value_freq = {}
        for idx, row in df_ordered.iterrows():
            for v in row.values:
                v_str = str(v)
                value_freq[v_str] = value_freq.get(v_str, 0) + 1
        
        prev_values = None
        for idx, row in df_ordered.iterrows():
            row_values = [str(v) for v in row.values]
            
            if prev_values is not None:
                perm = self._freq_perm(row_values, value_freq, length_exp)
            else:
                perm = row_values
            
            result_rows.append(perm)
            prev_values = perm
        
        return pd.DataFrame(result_rows, columns=cols_ordered)
    
    def _freq_perm(self, curr_values, value_freq, length_exp=2.37):
        result = sorted(
            curr_values,
            key=lambda v: (
                -(value_freq.get(str(v), 0) * (len(str(v)) ** length_exp)),
                -len(str(v)),
                str(v)
            )
        )
        return result
    
    def _smart_perm_freq(self, curr_values, prev_values, value_freq, length_exp=1.72):
        prev_set = set(prev_values)
        prev_list = list(prev_values)
        
        common = []
        unique = []
        
        for v in curr_values:
            if v in prev_set:
                common.append(v)
            else:
                unique.append(v)
        
        common_indexed = [
            (prev_list.index(v), -(value_freq.get(v, 0) * (len(str(v)) ** length_exp)), v)
            for v in common if v in prev_set
        ]
        common_indexed.sort(key=lambda x: (x[0], x[1]))
        common_sorted = [x[2] for x in common_indexed]
        
        unique_sorted = sorted(
            unique,
            key=lambda v: (
                -(value_freq.get(str(v), 0) * (len(str(v)) ** length_exp)),
                -len(str(v)),
                str(v)
            )
        )
        
        return common_sorted + unique_sorted