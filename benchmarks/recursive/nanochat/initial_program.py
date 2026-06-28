# P (reference): recursive nanochat vanilla transformer.
# This is an IMPORT-SAFE shim — importing it must NOT start training. It only
# points run_workload.py at the real training script, which is launched as a
# fresh subprocess (see run_workload.py).
import os

PROGRAM_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "recursive", "vanilla_transformer.py"
)
