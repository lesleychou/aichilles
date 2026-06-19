import json
import os


def run_workload(program_module, workload: dict):
    # Import inside function — harness exec's this code before _load_program adds the
    # cloudcast directory to sys.path; by the time run_workload() is *called*, it is set.
    from initial_program import make_nx_graph

    config_file = workload.get("config_file", "examples/config/intra_aws.json")
    # Walk up from program_module's location to find the app root (contains evaluator.py).
    # program_module.__file__ may be inside best/model/algo/ when running baselines.
    _d = os.path.dirname(os.path.abspath(program_module.__file__))
    while _d != os.path.dirname(_d):
        if os.path.exists(os.path.join(_d, "evaluator.py")):
            break
        _d = os.path.dirname(_d)
    app_dir = _d
    config_path = os.path.join(app_dir, config_file)
    with open(config_path) as f:
        config = json.load(f)

    source_node    = config["source_node"]
    dest_nodes     = config["dest_nodes"]
    num_partitions = workload.get("num_partitions", config.get("num_partitions", 10))

    num_vms = int(workload.get("num_vms", 2))
    G    = make_nx_graph(num_vms=num_vms)
    bc_t = program_module.search_algorithm(source_node, dest_nodes, G, num_partitions)

    # Return a JSON-serializable metric — total path hops across all (dst, partition) pairs.
    # Fewer hops → lower simulated cost → better program. This varies between programs
    # without requiring the full simulator.
    total_hops = 0
    for dst in dest_nodes:
        for p in range(num_partitions):
            path = bc_t.paths.get(dst, {}).get(str(p)) or bc_t.paths.get(dst, {}).get(p)
            if path:
                total_hops += len(path)

    return {
        "total_hops":     total_hops,
        "num_dsts":       len(dest_nodes),
        "num_partitions": num_partitions,
    }
