# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """
    Directed Steiner-like broadcast via hub-augmented metric-closure arborescence.
    - Filter to finite-cost edges and remove self/incoming-to-root edges.
    - Identify relay hubs that lie on many src->dst shortest paths to encourage
      early branching and prefix sharing across clouds.
    - Build a directed metric-closure over {src} ∪ dsts ∪ hubs and compute a
      minimum arborescence rooted at src.
    - Expand closure edges back to original paths to form a shared broadcast DAG,
      then reuse shared prefixes across all partitions to minimize total cost.
    """
    # Build a filtered working graph with finite costs
    h = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        c = data.get("cost", None)
        if u == v or c is None:
            continue
        h.add_edge(u, v, **data)
    # Disallow incoming edges to src for rooting
    h.remove_edges_from(list(h.in_edges(src)))

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Dijkstra cache
    sp_cache = {}
    def dijkstra_from(u):
        if u in sp_cache:
            return sp_cache[u]
        try:
            du, pu = nx.single_source_dijkstra(h, u, weight="cost")
        except Exception:
            du, pu = {}, {}
        sp_cache[u] = (du, pu)
        return du, pu

    # Discover relay hubs on many src->dst shortest paths
    du_src, pu_src = dijkstra_from(src)
    freq = {}
    for d in dsts:
        p = pu_src.get(d)
        if not p:
            try:
                p = nx.dijkstra_path(h, src, d, weight="cost")
            except Exception:
                p = None
        if not p:
            continue
        for node in p[1:-1]:
            if node not in dsts:
                freq[node] = freq.get(node, 0) + 1
    relay_cands = [n for n, _ in sorted(freq.items(), key=lambda x: -x[1])][:min(8, max(0, len(dsts) - 1))]

    # Terminals include src, all dsts, and top relay hubs
    terminals = [src] + [d for i, d in enumerate(dsts) if d not in dsts[:i]]
    for r in relay_cands:
        if r not in terminals:
            terminals.append(r)

    # All-pairs shortest paths among terminals
    dist = {}
    path_map = {}
    for u in terminals:
        du, pu = dijkstra_from(u)
        dist[u] = du
        path_map[u] = pu

    # Metric closure over terminals (no incoming edges to src)
    K = nx.DiGraph()
    K.add_nodes_from(terminals)
    for u in terminals:
        for v in terminals:
            if u == v or v == src:
                continue
            d = dist.get(u, {}).get(v, float("inf"))
            if d < float("inf"):
                K.add_edge(u, v, weight=d)

    # Minimum arborescence on closure; keep only edges reachable from src
    T = nx.DiGraph()
    try:
        T_full = nx.algorithms.tree.branchings.minimum_spanning_arborescence(K, attr="weight", default=float("inf"))
        reachable = {src}
        changed = True
        while changed:
            changed = False
            for a, b in T_full.edges():
                if a in reachable and b not in reachable:
                    reachable.add(b)
                    changed = True
        for a, b, data in T_full.edges(data=True):
            if a in reachable:
                T.add_edge(a, b, weight=data["weight"])
    except Exception:
        pass

    # Greedily attach any missing dsts to nearest terminal already connected
    covered = {src}
    if T.number_of_nodes() > 0:
        covered |= set(nx.descendants(T, src))
    missing = [v for v in dsts if v not in covered]
    while missing:
        best = (float("inf"), None, None)  # (d, u, v)
        cand_us = list(dict.fromkeys([src] + list(T.nodes())))
        for v in missing:
            for u in cand_us:
                d = dist.get(u, {}).get(v, float("inf"))
                if d < best[0]:
                    best = (d, u, v)
        if best[1] is None or best[0] == float("inf"):
            break
        _, u, v = best
        T.add_edge(u, v, weight=dist[u][v])
        covered = {src} | set(nx.descendants(T, src))
        missing = [x for x in dsts if x not in covered]

    # Expand closure edges to original-graph paths and unify into a broadcast DAG
    U = nx.DiGraph()
    if T.number_of_edges() == 0:
        # Fallback: greedy Steiner growth on original graph
        tree_nodes = {src}
        remaining = set(dsts)
        while remaining:
            best = (float("inf"), None, None, None)
            for u in list(tree_nodes):
                du, pu = dijkstra_from(u)
                for v in list(remaining):
                    if v in pu:
                        p = pu[v]
                        c = sum(h[p[i]][p[i + 1]]["cost"] for i in range(len(p) - 1))
                        if c < best[0]:
                            best = (c, u, v, p)
            if best[3] is None:
                break
            _, _, v, p = best
            for i in range(len(p) - 1):
                a, b = p[i], p[i + 1]
                U.add_edge(a, b, **h[a][b])
                tree_nodes.add(a); tree_nodes.add(b)
            remaining.discard(v)
    else:
        for (u, v) in T.edges():
            p = path_map.get(u, {}).get(v)
            if not p:
                try:
                    p = nx.dijkstra_path(h, u, v, weight="cost")
                except Exception:
                    p = None
            if p:
                for i in range(len(p) - 1):
                    a, b = p[i], p[i + 1]
                    U.add_edge(a, b, **h[a][b])

    # Emit per-dst, per-partition hop sequences along shared paths
    for dst in dsts:
        try:
            p = nx.shortest_path(U, src, dst, weight=None)
        except Exception:
            try:
                p = nx.dijkstra_path(h, src, dst, weight="cost")
            except Exception:
                p = None
        if not p or len(p) < 2:
            continue
        for j in range(num_partitions):
            for i in range(len(p) - 1):
                s, t = p[i], p[i + 1]
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
