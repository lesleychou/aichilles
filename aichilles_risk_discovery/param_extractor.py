"""
Deterministic param extraction from evaluator.py source via AST.

Public API:
    extract_params(app_dir: Path) -> dict
        Returns {"extracted": list[dict], "methods_used": list[str]}
        Each param dict has at minimum: name, type, source.
"""
import ast
import json
import sys
from pathlib import Path

# ── Shared helpers ─────────────────────────────────────────────────────────────

_PATH_SUFFIXES = ('.json', '.csv', '.py', '.txt', '.pkl', '.pt', '.npy', '.npz')


def _is_path_string(v) -> bool:
    """Hardcoded path filter — isolated here, used only in extractor 1 and 3."""
    return isinstance(v, str) and v.endswith(_PATH_SUFFIXES)


# ── Extractor 1: module-level constants + function defaults ────────────────────

def extract_constants(source: str) -> list[dict]:
    """
    AST walk for:
      - module-scope  NAME = <literal>
      - function default args  def fn(param=<literal>)

    Drops any string value that looks like a file path.
    Returns list of {name, type, value, source}.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: list[dict] = []

    for node in ast.iter_child_nodes(tree):
        # Pattern A: module-level assignment
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if not isinstance(node.value, ast.Constant):
                continue
            value = node.value.value
            if _is_path_string(value):
                continue
            results.append({
                "name": target.id,
                "type": type(value).__name__,
                "value": value,
                "source": "module_const",
            })

        # Pattern B: function default args
        elif isinstance(node, ast.FunctionDef):
            args = node.args.args
            defaults = node.args.defaults
            # defaults align to the *last* len(defaults) positional args
            offset = len(args) - len(defaults)
            for i, default in enumerate(defaults):
                if not isinstance(default, ast.Constant):
                    continue
                value = default.value
                if _is_path_string(value):
                    continue
                arg_name = args[offset + i].arg
                results.append({
                    "name": arg_name,
                    "type": type(value).__name__,
                    "value": value,
                    "source": "fn_default",
                })

    return results


# ── Extractor 2: random generator bounds ──────────────────────────────────────

_RANDINT_NAMES = frozenset({'randint', 'randrange'})
_CHOICE_NAMES  = frozenset({'choice'})
_UNIFORM_NAMES = frozenset({'uniform'})


def _resolve_call_name(node: ast.Call) -> str | None:
    """Return the bare function name (last attribute) of a Call node, or None."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def extract_generator_bounds(source: str) -> list[dict]:
    """
    AST walk for np.random.randint / random.randint / random.choice / np.random.uniform.
    Associates each call with its enclosing ast.Assign target for a name.
    Falls back to 'random_int_N' / 'random_choice_N' / 'random_float_N' if no enclosing assign.
    Returns list of {name, type, low, high, source} or {name, type, values, source}.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # Build parent map: id(child_node) -> parent_node
    parent_map: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node

    results: list[dict] = []
    seen: set[tuple] = set()          # (name, low, high) or (name, tuple(values))
    counters = {"int": 0, "float": 0, "choice": 0}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn_name = _resolve_call_name(node)
        if fn_name is None:
            continue

        # Retrieve variable name from enclosing assignment
        parent = parent_map.get(id(node))
        var_name: str | None = None
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
            if isinstance(parent.targets[0], ast.Name):
                var_name = parent.targets[0].id

        if fn_name in _RANDINT_NAMES:
            if len(node.args) >= 2 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[1], ast.Constant):
                lo, hi = int(node.args[0].value), int(node.args[1].value)
                name = var_name or f"random_int_{counters['int']}"
                key = (name, lo, hi)
                if key not in seen:
                    seen.add(key)
                    counters["int"] += 1
                    results.append({"name": name, "type": "int", "low": lo, "high": hi - 1, "source": "randint"})

        elif fn_name in _CHOICE_NAMES:
            if len(node.args) >= 1 and isinstance(node.args[0], ast.List):
                elts = [e.value for e in node.args[0].elts if isinstance(e, ast.Constant)]
                if elts:
                    name = var_name or f"random_choice_{counters['choice']}"
                    key = (name, tuple(elts))
                    if key not in seen:
                        seen.add(key)
                        counters["choice"] += 1
                        results.append({"name": name, "type": type(elts[0]).__name__, "values": elts, "source": "choice"})

        elif fn_name in _UNIFORM_NAMES:
            if len(node.args) >= 2 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[1], ast.Constant):
                lo, hi = float(node.args[0].value), float(node.args[1].value)
                name = var_name or f"random_float_{counters['float']}"
                key = (name, lo, hi)
                if key not in seen:
                    seen.add(key)
                    counters["float"] += 1
                    results.append({"name": name, "type": "float", "low": lo, "high": hi, "source": "uniform"})

    return results


# ── Extractor 3: function-body local constants and call kwargs ─────────────────

_SHORT_NAMES   = frozenset({'i', 'j', 'k', 'x', 'y', 't', 's', 'n', 'e'})
_TRIVIAL_INTS  = frozenset({-1, 0, 1})
_TRIVIAL_FLOATS = frozenset({0.0, 1.0, -1.0})


def _should_include(name: str, value, seen_names: set[str]) -> bool:
    if name in seen_names or name in _SHORT_NAMES:
        return False
    if _is_path_string(value):
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, int) and value in _TRIVIAL_INTS:
        return False
    if isinstance(value, float) and value in _TRIVIAL_FLOATS:
        return False
    return True


def extract_local_constants(source: str, known_names: set[str]) -> list[dict]:
    """
    Scan all ast.FunctionDef bodies for:
      A. name = <literal>  (local assignment, non-trivial, non-path)
      B. fn(..., kw=<literal>, ...)  (keyword arg in any call)

    known_names: names already captured by extractors 1+2 — skipped to avoid duplicates.
    Returns list of {name, type, value, source} where source is 'local_const' or 'fn_kwarg'.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: list[dict] = []
    seen_names = set(known_names)

    for fn_node in ast.walk(tree):
        if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for node in ast.walk(fn_node):
            # Pattern A: name = literal
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    if isinstance(node.value, ast.Constant):
                        name  = node.targets[0].id
                        value = node.value.value
                        if _should_include(name, value, seen_names):
                            seen_names.add(name)
                            results.append({"name": name, "type": type(value).__name__, "value": value, "source": "local_const"})

            # Pattern B: fn(..., kw=literal, ...)
            elif isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg is None:          # **kwargs spread
                        continue
                    if isinstance(kw.value, ast.Constant):
                        name  = kw.arg
                        value = kw.value.value
                        if _should_include(name, value, seen_names):
                            seen_names.add(name)
                            results.append({"name": name, "type": type(value).__name__, "value": value, "source": "fn_kwarg"})

    return results


# ── Extractor 4: sibling module data (fallback) ────────────────────────────────

def extract_module_data(initial_source: str, app_dir: Path) -> list[dict]:
    """
    Scan initial_program.py for 'from X import ...' where X.py exists in app_dir.
    Load X.py and find module-level string constants that JSON-parse to dicts.
    Records num_transactions and key pattern for workload-like structures.
    Returns list of {name, num_transactions, key_pattern, source}.
    """
    try:
        tree = ast.parse(initial_source)
    except SyntaxError:
        return []

    results: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_name = node.module
        if not module_name:
            continue
        sibling_path = app_dir / f"{module_name}.py"
        if not sibling_path.exists():
            continue

        try:
            sibling_source = sibling_path.read_text()
            sibling_tree = ast.parse(sibling_source)
        except (OSError, SyntaxError):
            continue

        for sibling_node in ast.iter_child_nodes(sibling_tree):
            if not isinstance(sibling_node, ast.Assign):
                continue
            if len(sibling_node.targets) != 1:
                continue
            if not isinstance(sibling_node.targets[0], ast.Name):
                continue
            if not isinstance(sibling_node.value, ast.Constant):
                continue

            name  = sibling_node.targets[0].id
            value = sibling_node.value.value

            if not isinstance(value, str):
                continue

            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                continue

            if not isinstance(parsed, dict):
                continue

            keys = list(parsed.keys())
            num_transactions = len(keys)
            # Detect "txn0", "txn1", ... pattern
            key_pattern = "txn{i}" if all(k.startswith("txn") for k in keys[:5]) else "unknown"

            results.append({
                "name": name,
                "num_transactions": num_transactions,
                "key_pattern": key_pattern,
                "source": f"module:{module_name}",
            })

    return results


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_params(app_dir: Path) -> dict:
    """
    Run all four extractors on evaluator.py (and initial_program.py for fallback).

    Returns:
        {
          "extracted": list[dict],     # all params found, deduplicated
          "methods_used": list[str],   # which extractors contributed
        }

    Never raises — all extractor failures are caught and logged to stderr.
    """
    evaluator_path = app_dir / "evaluator.py"
    initial_path   = app_dir / "initial_program.py"

    if not evaluator_path.exists():
        return {"extracted": [], "methods_used": []}

    source = evaluator_path.read_text()
    extracted: list[dict] = []
    methods_used: list[str] = []

    # ── Extractor 1: module constants + function defaults ──────────────────────
    try:
        consts = extract_constants(source)
        if consts:
            extracted.extend(consts)
            methods_used.append("module_constants")
    except Exception as exc:
        print(f"[param_extractor] extractor 1 failed: {exc}", file=sys.stderr)

    # ── Extractor 2: random generator bounds ───────────────────────────────────
    try:
        bounds = extract_generator_bounds(source)
        if bounds:
            extracted.extend(bounds)
            methods_used.append("generator_bounds")
    except Exception as exc:
        print(f"[param_extractor] extractor 2 failed: {exc}", file=sys.stderr)

    # ── Extractor 3: function-body locals + kwargs ─────────────────────────────
    known_names = {p["name"] for p in extracted}
    try:
        locals_ = extract_local_constants(source, known_names)
        if locals_:
            extracted.extend(locals_)
            methods_used.append("local_constants")
    except Exception as exc:
        print(f"[param_extractor] extractor 3 failed: {exc}", file=sys.stderr)

    # ── Extractor 4: sibling module data (fallback) ────────────────────────────
    # Triggered only when extractors 1+2 found nothing (extractor 3 noise excluded)
    e1_e2_count = sum(
        1 for p in extracted
        if p.get("source") in ("module_const", "fn_default", "randint", "randrange", "uniform", "choice")
    )
    if e1_e2_count == 0 and initial_path.exists():
        try:
            initial_source = initial_path.read_text()
            module_data = extract_module_data(initial_source, app_dir)
            if module_data:
                extracted.extend(module_data)
                methods_used.append("sibling_module")
        except Exception as exc:
            print(f"[param_extractor] extractor 4 failed: {exc}", file=sys.stderr)

    return {"extracted": extracted, "methods_used": methods_used}
