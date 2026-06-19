# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    
    # Create a composite weight that balances cost and throughput
    # Lower throughput edges get penalized to avoid bottlenecks
    for u, v, data in h.edges(data=True):
        base_cost = data.get('cost', 1.0)
        throughput = data.get('throughput', 1.0)
        # Penalize low-throughput edges slightly (10% factor)
        # This helps avoid congestion while keeping cost as primary metric
        throughput_penalty = 0.1 / max(throughput, 0.01)
        data['composite_weight'] = base_cost + throughput_penalty
    
    # Build an incremental Steiner tree to maximize edge reuse
    tree_nodes = {src}  # Nodes already in the multicast tree
    tree_edges = {}     # (s, t) -> edge data for edges in tree
    dst_paths = {}      # dst -> complete path from src to dst
    
    # Sort destinations by distance from source (closer destinations first)
    dst_distances = []
    for dst in dsts:
        try:
            dist = nx.dijkstra_path_length(h, src, dst, weight="composite_weight")
            dst_distances.append((dist, dst))
        except nx.NetworkXNoPath:
            dst_distances.append((float('inf'), dst))
    
    dst_distances.sort()
    
    # Incrementally build the tree by connecting each destination
    for _, dst in dst_distances:
        best_full_path = None
        best_new_cost = float('inf')  # Cost of edges not yet in tree (prefer reusing tree edges)
        best_total_cost = float('inf')
        
        # Try connecting from each node already in the tree
        for tree_node in tree_nodes:
            try:
                # Find path from this tree node to the destination
                path_segment = nx.dijkstra_path(h, tree_node, dst, weight="composite_weight")
                
                # Calculate the cost of new edges (not already in tree)
                new_edge_cost = 0
                total_edge_cost = 0
                for i in range(len(path_segment) - 1):
                    edge = (path_segment[i], path_segment[i + 1])
                    # Use original cost for comparison, not composite weight
                    edge_cost = G[path_segment[i]][path_segment[i + 1]]['cost']
                    total_edge_cost += edge_cost
                    if edge not in tree_edges:
                        new_edge_cost += edge_cost
                
                # Build complete path from source
                if tree_node == src:
                    full_path = path_segment
                    full_cost = total_edge_cost
                    new_cost = new_edge_cost
                else:
                    # Need to prepend the path from src to tree_node
                    if tree_node in dst_paths:
                        prefix = dst_paths[tree_node]
                    else:
                        # Find path through tree
                        try:
                            prefix = nx.dijkstra_path(h, src, tree_node, weight="composite_weight")
                        except nx.NetworkXNoPath:
                            continue
                    
                    # Merge paths (avoid duplicating the tree_node)
                    full_path = prefix[:-1] + path_segment
                    
                    # Calculate full cost from source using original costs
                    prefix_cost = sum(G[prefix[i]][prefix[i + 1]]['cost'] 
                                    for i in range(len(prefix) - 1))
                    full_cost = prefix_cost + total_edge_cost
                    new_cost = new_edge_cost  # Only count truly new edges
                
                # Prefer paths with more edge reuse (lower new_cost)
                # Break ties with total cost
                if (new_cost < best_new_cost or 
                    (new_cost == best_new_cost and full_cost < best_total_cost)):
                    best_new_cost = new_cost
                    best_total_cost = full_cost
                    best_full_path = full_path
                    
            except (nx.NetworkXNoPath, KeyError):
                continue
        
        # Fallback if no path found through tree
        if best_full_path is None:
            try:
                best_full_path = nx.dijkstra_path(h, src, dst, weight="composite_weight")
            except nx.NetworkXNoPath:
                if h.has_edge(src, dst):
                    best_full_path = [src, dst]
                else:
                    continue
        
        # Store the complete path and update the tree
        dst_paths[dst] = best_full_path
        tree_nodes.update(best_full_path)
        for i in range(len(best_full_path) - 1):
            edge = (best_full_path[i], best_full_path[i + 1])
            tree_edges[edge] = G[best_full_path[i]][best_full_path[i + 1]]
    
    sorted_dsts = dsts  # Process in order they were added to tree
    
    for dst in sorted_dsts:
        path = dst_paths.get(dst, [])
        if len(path) < 2:
            continue
        
        # All partitions follow the same path to enable multicast
        # The cost model accounts for shared edges across destinations
        for j in range(num_partitions):
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])
    
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
