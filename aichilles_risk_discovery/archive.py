"""
MAP-Elites archive and matrix V data structures for Agent 2.

Public API:
  coverage_to_vector(counts, n_lines) -> np.ndarray
  normalize_vector(v) -> np.ndarray
  compute_novelty(v, archive_vectors) -> float
  bucket_key(v, k) -> tuple[int, ...]
  MapElitesArchive   — per-signature sub-archives prevent correctness bugs
                       (delta=1.0) from evicting optimality seeds (delta≈0.05)
  MatrixV            — append-only log of all tried (c,w) pairs
"""
import random
from typing import Any

import numpy as np

_ALL_SIGNATURES = ["correctness", "scalab_time", "scalab_mem", "optimality"]


def coverage_to_vector(counts: dict[int, int], n_lines: int) -> np.ndarray:
    """
    Convert line-number → count dict to a dense vector of length n_lines.
    Line number l maps to index (l - 1). Out-of-range lines are ignored.
    """
    v = np.zeros(n_lines, dtype=float)
    for lineno, freq in counts.items():
        idx = lineno - 1
        if 0 <= idx < n_lines:
            v[idx] += freq
    return v


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize by L1 norm. Returns zero vector if sum is zero."""
    total = v.sum()
    if total == 0:
        return v.copy()
    return v / total


def compute_novelty(v: np.ndarray, archive_vectors: list[np.ndarray]) -> float:
    """
    Minimum L1 distance from v to all vectors in archive_vectors.
    Returns inf if archive is empty.
    """
    if not archive_vectors:
        return float("inf")
    return float(min(np.abs(v - a).sum() for a in archive_vectors))


def bucket_key(v: np.ndarray, k: int = 5) -> tuple[int, ...]:
    """
    Coarse hash: indices of the top-k most active elements, sorted ascending.
    Used as MAP-Elites bucket key.
    """
    n = min(k, int((v > 0).sum()))
    if n == 0:
        return ()
    indices = np.argsort(v)[::-1][:n]
    return tuple(sorted(indices.tolist()))


class MapElitesArchive:
    """
    Per-signature sub-archives prevent high-delta correctness bugs (delta=1.0)
    from permanently evicting low-delta optimality bugs (delta≈0.05-0.2) out of
    every coverage bucket.

    Each oracle signature maintains its own bucket map.  A global bucket map
    tracks all entries for novelty vector computation.

    Sampling with prefer_signature draws from that signature's sub-archive so
    confirmed optimality (or any under-represented) bugs can be used as seeds
    to find more of the same kind.

    Visit counts per cell penalise heavily-explored cells during sampling:
    weight_i = 1 / (visit_count_i + 1), so a cell sampled N times is ~N+1x
    less likely to be chosen than an unexplored cell.
    """

    def __init__(self, bucket_k: int = 5):
        self._bucket_k = bucket_k
        # Per-signature: sig -> {bucket_key -> entry}
        self._sig_buckets: dict[str, dict[tuple, dict]] = {s: {} for s in _ALL_SIGNATURES}
        self._sig_vectors: dict[str, dict[tuple, np.ndarray]] = {s: {} for s in _ALL_SIGNATURES}
        # Global: any entry, used for novelty vector set and fallback sampling
        self._global_buckets: dict[tuple, dict] = {}
        self._global_vectors: dict[tuple, np.ndarray] = {}
        # Visit counts: how many times each cell has been used as a seed
        self._visit_counts: dict[tuple, int] = {}

    def update(self, v_norm: np.ndarray, entry: dict) -> bool:
        """
        Insert entry into archive.
        - BUG entries go into each fired signature's sub-archive (best delta per bucket).
        - All entries go into the global archive (best delta per bucket).
        Returns True if any sub-archive or global was modified.
        """
        key = bucket_key(v_norm, self._bucket_k)
        modified = False

        for sig in entry.get("signatures", []):
            if sig not in self._sig_buckets:
                continue
            existing = self._sig_buckets[sig].get(key)
            if existing is None or entry.get("delta", 0.0) > existing.get("delta", 0.0):
                self._sig_buckets[sig][key] = entry
                self._sig_vectors[sig][key] = v_norm
                modified = True

        existing_global = self._global_buckets.get(key)
        if existing_global is None or entry.get("delta", 0.0) > existing_global.get("delta", 0.0):
            self._global_buckets[key] = entry
            self._global_vectors[key] = v_norm
            modified = True

        return modified

    def _weighted_sample(self, bucket_dict: dict[tuple, dict]) -> dict:
        """
        Sample one entry with probability inversely proportional to visit count.
        weight_i = 1 / (visits_i + 1) so unexplored cells are preferred.
        """
        keys = list(bucket_dict.keys())
        weights = [1.0 / (self._visit_counts.get(k, 0) + 1) for k in keys]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for k, w in zip(keys, weights):
            cumulative += w
            if r <= cumulative:
                self._visit_counts[k] = self._visit_counts.get(k, 0) + 1
                return bucket_dict[k]
        # fallback (floating-point edge case)
        k = keys[-1]
        self._visit_counts[k] = self._visit_counts.get(k, 0) + 1
        return bucket_dict[k]

    def sample(self, prefer_signature: str | None = None) -> dict | None:
        """
        Sample one seed entry with visit-count penalty (heavily explored cells
        are less likely to be chosen).
        - If prefer_signature is given and that sub-archive is non-empty,
          sample from known bugs of that signature (exploit confirmed seeds).
        - Otherwise sample from the global archive.
        """
        if prefer_signature and self._sig_buckets.get(prefer_signature):
            return self._weighted_sample(self._sig_buckets[prefer_signature])
        if not self._global_buckets:
            return None
        return self._weighted_sample(self._global_buckets)

    def all_vectors(self) -> list[np.ndarray]:
        """Return all stored normalized coverage vectors (global set, for novelty)."""
        return list(self._global_vectors.values())

    def size(self) -> int:
        return len(self._global_buckets)

    def bug_counts_by_signature(self) -> dict[str, int]:
        """Count confirmed bug entries per signature across all sub-archives."""
        return {sig: len(buckets) for sig, buckets in self._sig_buckets.items() if buckets}


class MatrixV:
    """
    Append-only log of all tried (c, w) pairs with their coverage and oracle results.
    Entries are kept in memory; caller is responsible for checkpointing to disk.
    """

    def __init__(self):
        self._entries: list[dict[str, Any]] = []

    def append(self, entry: dict) -> None:
        self._entries.append(entry)

    def count(self) -> int:
        return len(self._entries)

    def bug_count(self) -> int:
        return sum(1 for e in self._entries if e.get("label") == "BUG")

    def all_entries(self) -> list[dict]:
        return list(self._entries)

    def no_bug_baseline(self) -> np.ndarray | None:
        """
        Mean of normalized coverage vectors over all NO_BUG entries.
        Returns None if no NO_BUG entries exist yet.
        """
        vecs = [np.array(e["v"]) for e in self._entries
                if e.get("label") == "NO_BUG" and e.get("v")]
        if not vecs:
            return None
        return np.mean(vecs, axis=0)
