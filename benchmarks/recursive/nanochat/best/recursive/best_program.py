# P' (candidate): recursive nanochat optimized_from_karpathy.
# Import-safe shim (see initial_program.py). __file__ is at <app>/best/recursive/,
# so walk up three levels to the app root, then into recursive/.
import os

_APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROGRAM_SCRIPT = os.path.join(_APP_ROOT, "recursive", "optimized_from_karpathy.py")
