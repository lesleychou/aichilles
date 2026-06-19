def search_algorithm(src, dsts, G, num_partitions):
    import networkx as nx
    import time as time_module
    import sys
    
    start_time = time_module.time()
    
    TRANSFER_SIZE = 300
    VM_COST_PER_SEC = 0.00015
    VM_LIMIT = 2
    partition_size = TRANSFER_SIZE / num_partitions
    
    def get_provider(node):
        return node.split(':')[0]
    
    provider_egress = {'aws': 5, 'gcp': 7, 'azure': 16}
    provider_ingress = {'aws': 10, 'gcp': 16, 'azure': 16}
    
    def get_egress_limit(node):
        return provider_egress.get(get_provider(node), 5) * VM_LIMIT
    
    # Debug: Print graph info
    print(f"\n=== GRAPH DEBUG: src={src}, dsts={dsts}, num_partitions={num_partitions} ===")
    print(f"  Total nodes: {len(G.nodes())}, Total edges: {len(G.edges())}")
    
    # Check edge attributes
    sample_edges = list(G.edges(data=True))[:3]
    for u, v, d in sample_edges:
        print(f"  Sample edge {u}->{v}: {d}")
    
    # Check node attributes
    sample_nodes = list(G.nodes(data=True))[:3]
    for n, d in sample_nodes:
        print(f"  Sample node {n}: {d}")
    
    # Check provider limits from graph
    for n, d in G.nodes(data=True):
        if d:
            print(f"  Node with data {n}: {d}")
            break
    
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    
    dsts = list(dsts)
    terminals = set([src] + dsts)
    
    def compute_cost(paths_dict):
        tree_edges = set()
        for path in paths_dict.values():
            for i in range(len(path)-1):
                tree_edges.add((path[i], path[i+1]))
        
        egress = sum(G[u][v]['cost'] for u,v in tree_edges) * TRANSFER_SIZE
        
        node_out = {}
        for u, v in tree_edges:
            node_out.setdefault(u, set()).add(v)
        
        max_time = 0
        for u, v in tree_edges:
            t = num_partitions * partition_size * 8.0 / G[u][v]['throughput']
            max_time = max(max_time, t)
        
        for node, children in node_out.items():
            total_data = len(children) * num_partitions * partition_size
            t = total_data * 8.0 / get_egress_limit(node)
            max_time = max(max_time, t)
        
        active = set()
        for path in paths_dict.values():
            for n in path:
                active.add(n)
        
        instance = VM_COST_PER_SEC * max_time * len(active) * VM_LIMIT
        return egress + instance
    
    import random
    random.seed(42)
    
    best_paths = None
    best_cost = float('inf')
    
    # Hub-and-spoke
    for _, hub, _ in h.out_edges(src, data=True):
        paths = {}
        ok = True
        for dst in dsts:
            if hub == dst:
                paths[dst] = [src, dst]
                continue
            try:
                p = nx.dijkstra_path(h, hub, dst, weight='cost')
                paths[dst] = [src] + p
            except:
                ok = False
                break
        if ok:
            cost = compute_cost(paths)
            if cost < best_cost:
                best_cost = cost
                best_paths = paths
    
    # Two-level hub
    for _, hub1, _ in h.out_edges(src, data=True):
        for _, hub2, _ in h.out_edges(hub1, data=True):
            if hub2 == src:
                continue
            paths = {}
            ok = True
            for dst in dsts:
                if dst == hub1:
                    paths[dst] = [src, hub1]
                elif dst == hub2:
                    paths[dst] = [src, hub1, hub2]
                else:
                    try:
                        p = nx.dijkstra_path(h, hub2, dst, weight='cost')
                        paths[dst] = [src, hub1, hub2] + p[1:]
                    except:
                        ok = False
                        break
            if ok:
                cost = compute_cost(paths)
                if cost < best_cost:
                    best_cost = cost
                    best_paths = paths
    
    # Greedy Steiner
    for trial in range(30):
        if time_module.time() - start_time > 15:
            break
        tree = nx.DiGraph()
        tree_nodes = set([src])
        sp = {}
        rem = list(dsts)
        if trial == 0:
            rem.sort(key=lambda d: nx.dijkstra_path_length(h, src, d, weight='cost'))
        elif trial == 1:
            rem.sort(key=lambda d: -nx.dijkstra_path_length(h, src, d, weight='cost'))
        else:
            random.shuffle(rem)
        
        for dst in rem:
            bp = None
            bnc = float('inf')
            for tn in list(tree_nodes):
                try:
                    if tn == src:
                        p = nx.dijkstra_path(h, src, dst, weight='cost')
                    else:
                        tp = nx.dijkstra_path(h, tn, dst, weight='cost')
                        s2t = nx.shortest_path(tree, src, tn)
                        p = s2t + tp[1:]
                    nc = sum(h[p[i]][p[i+1]]['cost'] for i in range(len(p)-1) if not tree.has_edge(p[i], p[i+1]))
                    if nc < bnc:
                        bnc = nc
                        bp = p
                except:
                    continue
            if bp is None:
                bp = nx.dijkstra_path(h, src, dst, weight='cost')
            sp[dst] = bp
            for i in range(len(bp)-1):
                tree.add_edge(bp[i], bp[i+1], **h[bp[i]][bp[i+1]])
                tree_nodes.update([bp[i], bp[i+1]])
        
        cost = compute_cost(sp)
        if cost < best_cost:
            best_cost = cost
            best_paths = sp
    
    print(f"  Heuristic best: ${best_cost:.2f}")
    
    # Print tree structure
    tree_edges = set()
    for path in best_paths.values():
        for i in range(len(path)-1):
            tree_edges.add((path[i], path[i+1]))
    print(f"  Tree edges ({len(tree_edges)}):")
    total_edge_cost = 0
    for u, v in sorted(tree_edges):
        print(f"    {u} -> {v}: cost={G[u][v]['cost']}, tp={G[u][v]['throughput']}")
        total_edge_cost += G[u][v]['cost']
    print(f"  Total edge cost/GB: ${total_edge_cost:.4f}")
    print(f"  Egress = ${total_edge_cost * TRANSFER_SIZE:.2f}")
    
    # ========================================
    # MILP
    # ========================================
    from pulp import (LpProblem, LpMinimize, LpVariable, lpSum, LpBinary,
                      LpContinuous, PULP_CBC_CMD, value)
    
    # Use full graph for MILP
    nodes = list(h.nodes())
    edges = list(h.edges())
    out_idx = {}
    in_idx = {}
    for u, v in edges:
        out_idx.setdefault(u, []).append((u, v))
        in_idx.setdefault(v, []).append((u, v))
    
    elapsed = time_module.time() - start_time
    tl = max(10, 105 - elapsed)
    
    prob = LpProblem("broadcast", LpMinimize)
    
    x = {(u,v): LpVariable(f"x_{u}_{v}", cat=LpBinary) for u, v in edges}
    f_vars = {(dst,u,v): LpVariable(f"f_{dst}_{u}_{v}", lowBound=0, upBound=1)
         for dst in dsts for u, v in edges}
    T = LpVariable("T", lowBound=0)
    
    for dst in dsts:
        for node in nodes:
            infl = lpSum(f_vars.get((dst,u,v), 0) for u, v in in_idx.get(node, []))
            outfl = lpSum(f_vars.get((dst,u,v), 0) for u, v in out_idx.get(node, []))
            if node == src:
                prob += outfl - infl == 1
            elif node == dst:
                prob += infl - outfl == 1
            else:
                prob += infl == outfl
    
    for dst in dsts:
        for u, v in edges:
            prob += f_vars[(dst,u,v)] <= x[(u,v)]
    
    egress_expr = lpSum(x[(u,v)] * TRANSFER_SIZE * h[u][v]['cost'] for u, v in edges)
    
    for node in nodes:
        oe = out_idx.get(node, [])
        if oe:
            num_out = lpSum(x[(u,v)] for u, v in oe)
            prob += T >= num_out * TRANSFER_SIZE * 8.0 / get_egress_limit(node)
    
    for u, v in edges:
        tp = h[u][v]['throughput']
        prob += T >= x[(u,v)] * TRANSFER_SIZE * 8.0 / tp
    
    active_est = len(terminals) + 3
    instance_expr = VM_COST_PER_SEC * T * active_est * VM_LIMIT
    prob += egress_expr + instance_expr
    
    # Warm-start with heuristic
    for u, v in edges:
        if (u, v) in tree_edges:
            x[(u,v)].setInitialValue(1)
        else:
            x[(u,v)].setInitialValue(0)
    
    for dst in dsts:
        path = best_paths[dst]
        path_edges = set()
        for i in range(len(path)-1):
            path_edges.add((path[i], path[i+1]))
        for u, v in edges:
            if (u, v) in path_edges:
                f_vars[(dst,u,v)].setInitialValue(1)
            else:
                f_vars[(dst,u,v)].setInitialValue(0)
    
    solver = PULP_CBC_CMD(timeLimit=int(tl), msg=0, warmStart=True)
    prob.solve(solver)
    
    print(f"  MILP status={prob.status}, obj={value(prob.objective):.2f}")
    
    if prob.status in [1, -1]:
        tree_g = nx.DiGraph()
        for u, v in edges:
            xv = value(x[(u,v)])
            if xv is not None and xv > 0.5:
                tree_g.add_edge(u, v, **h[u][v])
        
        milp_paths = {}
        ok = True
        for dst in dsts:
            try:
                path = nx.shortest_path(tree_g, src, dst)
                for i in range(len(path)-1):
                    if not G.has_edge(path[i], path[i+1]):
                        raise ValueError()
                milp_paths[dst] = path
            except:
                ok = False
                break
        
        if ok:
            milp_cost = compute_cost(milp_paths)
            print(f"  MILP cost: ${milp_cost:.2f}")
            if milp_cost < best_cost:
                best_paths = milp_paths
                best_cost = milp_cost
    
    print(f"  FINAL cost: ${best_cost:.2f}")
    
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    for dst in dsts:
        path = best_paths[dst]
        for k in range(num_partitions):
            for i in range(len(path)-1):
                bc_topology.append_dst_partition_path(dst, k, [path[i], path[i+1], G[path[i]][path[i+1]]])
    
    return bc_topology