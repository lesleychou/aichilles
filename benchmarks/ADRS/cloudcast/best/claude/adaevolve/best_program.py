# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """
    Greedy set cover with path bundling for optimal broadcast routing.
    
    Strategy:
    1. Compute k shortest paths for each destination (k=6 for diversity)
    2. Calculate path costs and potential coverage (# of partitions per dst)
    3. Greedily select paths that maximize coverage per unit cost
    4. Bundle multiple partitions onto the same cost-effective path
    5. Continue until all (dst, partition) pairs are covered
    
    Key insight: Treat routing as a discrete optimization problem where we select
    a minimal set of paths to cover all requirements. This naturally bundles partitions
    onto cost-effective paths and exploits path sharing across destinations.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    
    # Step 1: Compute k shortest paths for each destination
    k_paths = 6  # Limit to avoid exponential enumeration
    dst_paths = {}  # dst -> list of (path, cost)
    
    for dst in dsts:
        paths_list = []
        try:
            # Get shortest path first (guaranteed to exist if reachable)
            shortest = nx.dijkstra_path(h, src, dst, weight='cost')
            shortest_cost = sum(G[shortest[i]][shortest[i+1]]['cost'] for i in range(len(shortest)-1))
            paths_list.append((shortest, shortest_cost))
            
            # Find alternative paths using shortest_simple_paths
            # Set cost cutoff to 2x shortest to avoid very expensive alternatives
            cost_cutoff = shortest_cost * 2.0
            
            path_generator = nx.shortest_simple_paths(h, src, dst, weight='cost')
            next(path_generator)  # Skip first (already have shortest)
            
            for path in path_generator:
                if len(paths_list) >= k_paths:
                    break
                    
                # Calculate actual cost
                path_cost = sum(G[path[i]][path[i+1]]['cost'] for i in range(len(path)-1))
                
                # Only add if within cost cutoff
                if path_cost <= cost_cutoff:
                    paths_list.append((path, path_cost))
                    
        except (nx.NetworkXNoPath, StopIteration):
            # If no path or generator exhausted, check direct connection
            if not paths_list and h.has_edge(src, dst):
                paths_list.append(([src, dst], G[src][dst]['cost']))
        
        dst_paths[dst] = paths_list
    
    # Step 2: Greedy set cover - select paths to minimize total cost
    # Track which (dst, partition) pairs are covered
    uncovered = {(dst, p) for dst in dsts for p in range(num_partitions)}
    path_assignments = {dst: [None] * num_partitions for dst in dsts}  # dst -> partition -> path_idx
    
    while uncovered:
        best_path = None
        best_dst = None
        best_ratio = float('inf')
        best_coverage = []
        
        # For each destination and each of its paths
        for dst in dsts:
            dst_uncovered = [(d, p) for (d, p) in uncovered if d == dst]
            if not dst_uncovered:
                continue
                
            for path_idx, (path, cost) in enumerate(dst_paths[dst]):
                # How many uncovered partitions can this path cover?
                coverage = dst_uncovered
                
                if not coverage:
                    continue
                
                # Calculate cost per coverage ratio
                # Paths already used have 0 incremental cost (we pay once per path)
                is_used = any(path_assignments[dst][p] == path_idx for p in range(num_partitions))
                effective_cost = 0.0 if is_used else cost
                
                # Ratio = cost / coverage (lower is better)
                ratio = effective_cost / len(coverage) if coverage else float('inf')
                
                # Select path with best ratio (prefer already-used paths with 0 cost)
                if ratio < best_ratio or (ratio == best_ratio and len(coverage) > len(best_coverage)):
                    best_ratio = ratio
                    best_path = path_idx
                    best_dst = dst
                    best_coverage = coverage
        
        # Assign all uncovered partitions of best_dst to best_path
        if best_path is not None and best_dst is not None:
            for dst, partition in best_coverage:
                path_assignments[dst][partition] = best_path
                uncovered.remove((dst, partition))
        else:
            # No valid path found - shouldn't happen but handle gracefully
            # Assign remaining to first available path
            for dst, partition in list(uncovered):
                if dst_paths[dst]:
                    path_assignments[dst][partition] = 0
                    uncovered.remove((dst, partition))
            break
    
    # Step 3: Build topology from assignments
    for dst in dsts:
        for partition_idx in range(num_partitions):
            path_idx = path_assignments[dst][partition_idx]
            if path_idx is not None and path_idx < len(dst_paths[dst]):
                path, _ = dst_paths[dst][path_idx]
                for i in range(len(path) - 1):
                    s, t = path[i], path[i + 1]
                    bc_topology.append_dst_partition_path(dst, partition_idx, [s, t, G[s][t]])
    
    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        print(f"now the set path is: {self.paths}")
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst
        """
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """
        Append path for partition = partition to reach dst
        """
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    """
    Default graph with capacity constraints and cost info
    nodes: regions, edges: links
    per edge:
        throughput: max tput achievable (gbps)
        cost: $/GB
        flow: actual flow (gbps), must be < throughput, default = 0
    """
    # Use relative path from this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if cost_path is None:
        cost = pd.read_csv(os.path.join(current_dir, "profiles/cost.csv"))
    else:
        cost = pd.read_csv(cost_path)

    if throughput_path is None:
        throughput = pd.read_csv(os.path.join(current_dir, "profiles/throughput.csv"))
    else:
        throughput = pd.read_csv(throughput_path)

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        if row["src_region"] == row["dst_region"]:
            continue
        G.add_edge(row["src_region"], row["dst_region"], cost=None, throughput=num_vms * row["throughput_sent"] / 1e9)

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            G[row["src"]][row["dest"]]["cost"] = row["cost"]

    # some pairs not in the cost grid
    no_cost_pairs = []
    for edge in G.edges.data():
        src, dst = edge[0], edge[1]
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((src, dst))
    print("Unable to get costs for: ", no_cost_pairs)

    return G


# EVOLVE-BLOCK-END

# Helper functions that won't be evolved
def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance"""
    return BroadCastTopology(src, dsts, num_partitions)

def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)
