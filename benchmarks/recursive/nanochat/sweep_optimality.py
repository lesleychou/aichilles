#!/usr/bin/env python3
"""
Direct P-vs-P' bpb sweep -> optimality regression CURVE. NO LLM, NO oracle, NO search.

Unlike plot_bugs.py (which only re-plots witnesses AIChilles already found, so it's
stuck at however many it discovered), this trains the reference P (vanilla) and a
chosen candidate P' on a hand-picked GRID of configs along ONE axis, and saves a
curve of val_bpb vs that axis for both — so you can SEE where P' crosses above P.

Run from the app root (benchmarks/recursive/nanochat), on the GPU box:

  export DATA_DIR=/home/ubuntu/data
  # vanilla-optimized candidate, sweep time_budget at the regression corner:
  AICHILLES_EAGER=1 AICHILLES_EVAL_TOKENS=2097152 python sweep_optimality.py \
      --candidate best/recursive_vanilla/best_program.py \
      --axis time_budget --values 20 40 80 160 300 \
      --seq_len 256 --depth 4 --device_batch_size 80 --n_seeds 3 \
      --out vanilla_opt_regression.png

  # or sweep seq_len instead:
  ... --axis seq_len --values 256 512 1024 2048 --time_budget 60

AICHILLES_EAGER + short EVAL_TOKENS inflate absolute bpb; the RELATIVE P-vs-P' gap
is what matters. Confirm any crossover faithfully (drop AICHILLES_EAGER, full eval).
"""
import argparse
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # so `import run_workload` and `from lib import` resolve

import run_workload as rw  # the app's fixed differential runner

_AXES = ["time_budget", "seq_len", "depth", "device_batch_size"]


def _load(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # import-safe: defines classes, does NOT train
    return mod


def _run(program, w):
    """Full run_workload output dict for `program` on `w`; None if it crashed."""
    try:
        return rw.run_workload(program, w)
    except Exception as exc:
        sys.stderr.write(f"    crash on {w}: {str(exc)[-200:]}\n")
        return None


def _mean_bpb(out):
    """Pull mean val_bpb from a run_workload output dict; None if missing."""
    if not out:
        return None
    try:
        return float(out["val_bpb_mean"])
    except (KeyError, TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default="initial_program.py",
                    help="P (baseline) program file (default: vanilla initial_program.py)")
    ap.add_argument("--candidate", default="best/recursive_vanilla/best_program.py",
                    help="P' (evolved) program file (default: vanilla-optimized)")
    ap.add_argument("--axis", default="time_budget", choices=_AXES,
                    help="workload param to sweep on the x-axis")
    ap.add_argument("--values", type=int, nargs="+", default=None,
                    help="x-axis values to sweep (default depends on --axis)")
    # Fixed (non-axis) workload params — defaults sit at the known regression corner.
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device_batch_size", type=int, default=80)
    ap.add_argument("--time_budget", type=int, default=60)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--out", default="optimality_regression.png")
    args = ap.parse_args()

    default_values = {
        "time_budget":       [20, 40, 80, 160, 300],
        "seq_len":           [256, 512, 1024, 2048],
        "depth":             [4, 5, 6, 7, 8],
        "device_batch_size": [8, 16, 32, 64, 80],
    }
    values = args.values or default_values[args.axis]

    ref_path = os.path.join(HERE, args.reference)
    cand_path = os.path.join(HERE, args.candidate)
    p = _load(ref_path)
    pp = _load(cand_path)

    fixed = {"seed": args.seed, "seq_len": args.seq_len, "depth": args.depth,
             "device_batch_size": args.device_batch_size, "time_budget": args.time_budget,
             "n_seeds": args.n_seeds}
    fixed.pop(args.axis, None)  # the axis param is swept, not fixed

    print(f"P  = {ref_path}")
    print(f"P' = {cand_path}")
    print(f"sweep {args.axis} over {values} | fixed={fixed} | "
          f"eager={os.environ.get('AICHILLES_EAGER','0')}\n")
    print(f"{args.axis:>16} | {'P bpb':>8} {'Pp bpb':>8} {'gap':>8}  verdict")
    print("-" * 60)

    out_base = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    out_base = os.path.splitext(out_base)[0]
    json_path, csv_path = out_base + ".json", out_base + ".csv"
    meta = {"reference": ref_path, "candidate": cand_path, "axis": args.axis,
            "values": values, "fixed": fixed, "n_seeds": args.n_seeds,
            "eager": os.environ.get("AICHILLES_EAGER", "0"),
            "eval_tokens": os.environ.get("AICHILLES_EVAL_TOKENS", "default")}

    def _save_raw(records):
        # Rewritten after EACH config so an interrupted (multi-hour) run keeps its
        # partial results. JSON holds the full run_workload output (incl. per-seed
        # val_bpb_seeds, std, num_steps, peak_vram_mb, training_seconds) for P and P'.
        with open(json_path, "w") as f:
            json.dump({**meta, "records": records}, f, indent=2)
        with open(csv_path, "w") as f:
            f.write("value,P_bpb,Pp_bpb,gap,P_steps,Pp_steps,P_std,Pp_std,"
                    "P_seconds,Pp_seconds\n")
            for r in records:
                op, opp = r["P"] or {}, r["P_prime"] or {}
                f.write(f'{r["value"]},{r["P_bpb"]},{r["Pp_bpb"]},{r["gap"]},'
                        f'{op.get("num_steps","")},{opp.get("num_steps","")},'
                        f'{op.get("val_bpb_std","")},{opp.get("val_bpb_std","")},'
                        f'{op.get("training_seconds","")},{opp.get("training_seconds","")}\n')

    records, xs, ys_p, ys_pp = [], [], [], []
    for v in values:
        w = {**fixed, args.axis: v}
        out_p, out_pp = _run(p, w), _run(pp, w)
        bpb_p, bpb_pp = _mean_bpb(out_p), _mean_bpb(out_pp)
        gap = (bpb_pp - bpb_p) if (bpb_p is not None and bpb_pp is not None) else None
        records.append({"value": v, "workload": w, "P_bpb": bpb_p, "Pp_bpb": bpb_pp,
                        "gap": gap, "P": out_p, "P_prime": out_pp})
        _save_raw(records)  # persist after every config
        if bpb_p is None or bpb_pp is None:
            verdict = "CRASH (correctness, not optimality)"
            print(f"{v:>16} | {'crash' if bpb_p is None else f'{bpb_p:.4f}':>8} "
                  f"{'crash' if bpb_pp is None else f'{bpb_pp:.4f}':>8} {'n/a':>8}  {verdict}")
            continue
        rel = gap / max(bpb_p, bpb_pp)
        verdict = (f"REGRESSION (P' worse {rel*100:.1f}%)" if gap > 0 else "P' better")
        print(f"{v:>16} | {bpb_p:>8.4f} {bpb_pp:>8.4f} {gap:>+8.4f}  {verdict}")
        xs.append(v); ys_p.append(bpb_p); ys_pp.append(bpb_pp)

    print(f"\nSaved raw results: {json_path}\n                   {csv_path}")
    if not xs:
        sys.exit("No non-crash points to plot.")

    # Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys_p,  "-o", color="steelblue", label="P (vanilla baseline)", linewidth=1.5)
    ax.plot(xs, ys_pp, "-o", color="firebrick", label=f"P' ({os.path.basename(os.path.dirname(cand_path))})", linewidth=1.5)
    ax.fill_between(xs, ys_p, ys_pp,
                    where=[pp_v > p_v for p_v, pp_v in zip(ys_p, ys_pp)],
                    color="firebrick", alpha=0.12, label="P' regression (worse bpb)")
    ax.set_xlabel(args.axis)
    ax.set_ylabel("val_bpb  (lower = better)")
    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items() if k != "n_seeds")
    ax.set_title(f"Optimality regression sweep  (n_seeds={args.n_seeds})\n{fixed_str}",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    fig.savefig(out, dpi=150)
    print(f"\nSaved curve: {out}")


if __name__ == "__main__":
    main()
