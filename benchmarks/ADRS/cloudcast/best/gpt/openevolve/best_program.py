# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    # Build a clean directed graph with valid costs only, no self-loops
    h = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if u == v or d.get("cost") is None:
            continue
        h.add_edge(u, v, **d)
    # Avoid cycles entering src
    h.remove_edges_from(list(h.in_edges(src)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Build an undirected cost graph to find a shared low-cost broadcast skeleton
    Hu = nx.Graph()
    for u, v, d in h.edges(data=True):
        c = d.get("cost")
        if c is None:
            continue
        if Hu.has_edge(u, v):
            if c < Hu[u][v]["cost"]:
                Hu[u][v]["cost"] = c
                Hu[u][v]["throughput"] = d.get("throughput")
        else:
            Hu.add_edge(u, v, cost=c, throughput=d.get("throughput"))

    terminals = set([src] + list(dsts))
    steiner_edges = set()
    try:
        from networkx.algorithms.approximation import steiner_tree as _steiner
        T = _steiner(Hu, terminals, weight="cost")
        steiner_edges = {frozenset((u, v)) for u, v in T.edges()}
    except Exception:
        # Fallback: union of individual shortest paths in Hu
        for d in dsts:
            try:
                p = nx.shortest_path(Hu, src, d, weight="cost")
                for i in range(len(p) - 1):
                    steiner_edges.add(frozenset((p[i], p[i + 1])))
            except Exception:
                continue

    # Build directed trunk-aware paths using the undirected Steiner tree as a shared backbone
    def trunk_path(dst):
        try:
            # Build a backbone graph from steiner_edges; fallback to Hu if empty
            if steiner_edges:
                Tg = nx.Graph()
                for e in steiner_edges:
                    u, v = tuple(e)
                    Tg.add_edge(u, v, **Hu[u][v])
            else:
                Tg = Hu
            t_nodes = nx.shortest_path(Tg, src, dst, weight="cost")
        except Exception:
            return None
        segs = []
        for i in range(len(t_nodes) - 1):
            a, b = t_nodes[i], t_nodes[i + 1]
            # Map each backbone hop to a feasible directed subpath in h
            try:
                sp = nx.dijkstra_path(h, a, b, weight="cost")
            except Exception:
                return None
            for k in range(len(sp) - 1):
                segs.append((sp[k], sp[k + 1]))
        return segs

    # Prefer the shared backbone; fall back to global cheapest route if needed
    for dst in dsts:
        segs = trunk_path(dst)
        if not segs:
            try:
                p = nx.dijkstra_path(h, src, dst, weight="cost")
                segs = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
            except Exception:
                continue
        for (s, t) in segs:
            edata = G[s][t] if s in G and t in G[s] else h[s][t]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, edata])

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
