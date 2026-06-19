import random as _random


def run_workload(program_module, workload: dict):
    gpu_num        = int(workload.get("gpu_num", 5))
    seed           = int(workload.get("seed", 42))
    model_size_max = int(workload.get("model_size_max", 8))

    # Generate a deterministic list of Model objects from gpu_num + seed,
    # mirroring generate_test_gpu_models() in evaluator.py.
    from evaluator import Model
    rng = _random.Random(seed)
    num_models = gpu_num * 2
    models = []
    for i in range(num_models):
        model_size = rng.randint(1, model_size_max)
        req_rate   = rng.randint(1, 100)
        slo        = rng.randint(1, 10)
        models.append(Model(
            model_name=f"model_{i}",
            model_size=model_size,
            req_rate=req_rate,
            slo=slo,
            cur_gpu_id=i % gpu_num,
        ))

    result = program_module.compute_model_placement(gpu_num=gpu_num, models=models)

    # Return a JSON-serializable metric — load balance across GPUs.
    # compute_model_placement returns {gpu_id: [Model, ...]}; Model objects can't be
    # pickled across the harness process boundary, so extract numeric summaries only.
    if not result:
        return {"balance_score": 0.0, "num_gpus_used": 0, "total_models": 0}
    gpu_loads = [sum(m.req_rate for m in v) for v in result.values()]
    max_load  = max(gpu_loads)
    avg_load  = sum(gpu_loads) / len(gpu_loads)
    balance   = avg_load / max_load if max_load > 0 else 1.0
    return {
        "balance_score":  balance,
        "num_gpus_used":  len(result),
        "total_models":   sum(len(v) for v in result.values()),
    }
