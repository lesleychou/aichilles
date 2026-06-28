#!/usr/bin/env python3
"""
Transform a canonical (train-on-import) nanochat source into an IMPORT-SAFE,
self-contained program file that AIChilles can import without training and run
as a subprocess.

Two edits, both string-marker based (no hardcoded line numbers):
  1. Drop the dead flash-attn-4 custom-op block (`cap = get_device_capability()`
     .. up to `from lib import`). The SDPA override right after `from lib import`
     already replaces flash_attn_func, so the FA4 block is dead code AND the only
     thing that inits CUDA at import — removing it makes the harness import cheap.
  2. Guard the train tail (`t_start = time.time()` .. EOF) behind
     `if __name__ == "__main__" or os.environ.get("AICHILLES_RUN") == "1":` so the
     harness can import the module (defs only) without training. run_workload runs
     the file as a subprocess with AICHILLES_RUN=1 to actually train.

Usage: python _make_import_safe.py <src.py> <dst.py>
"""
import sys

GUARD = 'if __name__ == "__main__" or os.environ.get("AICHILLES_RUN") == "1":'


def transform(src_text: str) -> str:
    lines = src_text.splitlines()

    # locate markers
    cap_i = next(i for i, l in enumerate(lines)
                 if l.strip().startswith("cap = torch.cuda.get_device_capability()"))
    lib_i = next(i for i, l in enumerate(lines)
                 if l.strip().startswith("from lib import"))
    tstart_i = next(i for i, l in enumerate(lines)
                    if l.strip().startswith("t_start = time.time()"))

    assert cap_i < lib_i < tstart_i, "unexpected marker order"

    head_before_fa4 = lines[:cap_i]
    # everything from `from lib import` (inclusive) up to (not incl.) the train tail
    head_after_fa4 = lines[lib_i:tstart_i]
    tail = lines[tstart_i:]

    # Safety: indenting a MULTI-line string literal would corrupt its contents.
    # Bracket/paren continuations and single-line docstrings are fine to indent;
    # only triple-quoted strings that SPAN lines are dangerous. A line that opens
    # or closes a multiline triple-quote has an ODD count of that quote; a
    # self-contained single-line docstring has an even count. Fail loudly on odd.
    for q in ('"""', "'''"):
        for l in tail:
            if l.count(q) % 2 == 1:
                raise AssertionError(
                    f"tail line opens/closes a multiline {q} string; indentation "
                    f"would corrupt it: {l!r}"
                )

    fa4_note = [
        "# NOTE: flash-attn-4 custom-op block removed (dead code: the SDPA override",
        "#       below replaces flash_attn_func, and FA4 is broken on this stack).",
        "#       Removing it also keeps this module's IMPORT cheap (no CUDA init),",
        "#       which matters because AIChilles imports it on every oracle call.",
    ]

    indented_tail = ["    " + l if l.strip() else "" for l in tail]

    out = (
        head_before_fa4
        + fa4_note
        + [""]
        + head_after_fa4
        + ["", "", "# --- Train block: runs ONLY as a subprocess (AICHILLES_RUN=1) or `python <file>`.",
           "#     Imported (by the AIChilles harness / Agent 3) it is skipped, so import is",
           "#     side-effect free: no tokenizer load, no model build, no training. ---",
           GUARD]
        + indented_tail
        + [""]
    )
    return "\n".join(out)


if __name__ == "__main__":
    src_path, dst_path = sys.argv[1], sys.argv[2]
    with open(src_path) as f:
        result = transform(f.read())
    # sanity: must be valid Python
    compile(result, dst_path, "exec")
    with open(dst_path, "w") as f:
        f.write(result)
    print(f"wrote import-safe {dst_path} ({len(result.splitlines())} lines)")
