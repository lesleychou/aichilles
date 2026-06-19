# aichilles_risk_discovery/agent1_infer.py
"""
Agent 1: infer Grammar_config, Grammar_workload, and generate_workload() for an ADRS app.

run_agent1(app_dir, best_program_path, results_dir, client) -> dict
  Writes grammar.json and generate_workload.py to results_dir.
  Returns {"grammar": {...}, "code": "..."}
"""
import json
import sys
import textwrap
from pathlib import Path

import anthropic

from harness import run_one
from schema_infer import infer_schema
from utils import extract_code_block, MODEL

MAX_RETRIES = 3


def _validate_generate_workload(code: str, app_dir: Path, initial_path: Path) -> tuple[bool, str]:
    """
    Execute generate_workload(grammar={}) from the LLM-generated code, then
    run the app's fixed run_workload.py against initial_program to confirm
    the sampled workload is valid.
    Returns (success, error_message).
    """
    try:
        ns = {}
        exec(code, ns)  # noqa: S102
        if "generate_workload" not in ns:
            return False, "generate_workload() not defined"
        cw = ns["generate_workload"]({})
        if not isinstance(cw, dict) or "c" not in cw or "w" not in cw:
            return False, f"generate_workload() must return dict with 'c' and 'w' keys, got: {cw}"
    except Exception as exc:
        return False, f"exec/generate_workload failed: {exc}"

    # Combine app's fixed run_workload.py with the generated generate_workload
    rw_path = app_dir / "run_workload.py"
    if not rw_path.exists():
        return False, f"app run_workload.py not found at {rw_path}"
    combined_code = rw_path.read_text() + "\n\n" + code

    result = run_one(str(initial_path), combined_code, cw["w"], timeout=30, collect_coverage=False)
    if result["error"] and result["error"] != "timeout":
        return False, f"run_workload failed on initial_program: {result['error']}"
    return True, ""


def run_agent1(
    app_dir: Path,
    best_program_path: Path,
    results_dir: Path,
    client: anthropic.Anthropic,
) -> dict:
    """
    Run Agent 1 for the given app. Writes grammar.json and generate_workload.py.
    Returns the grammar dict and the generated code.
    """
    print(f"[agent1] inferring schema for {app_dir.name}")
    grammar = infer_schema(app_dir, client)

    # Ask LLM for generate_workload code only.
    # run_workload is the fixed per-app file — Agent 1 must not write it.
    grammar_json = json.dumps(grammar, indent=2)
    rw_path = app_dir / "run_workload.py"
    run_workload_source = rw_path.read_text() if rw_path.exists() else ""

    code_prompt = textwrap.dedent(f"""
        The app already has a fixed run_workload function (do NOT rewrite it):

        ```python
        {run_workload_source}
        ```

        Your job is to write ONLY generate_workload(grammar: dict) -> dict.
        It must sample one valid workload and return {{"c": {{}}, "w": {{...}}}}
        where "w" contains exactly the keys that run_workload reads from
        workload.get(...) — use the same key names shown above.
        ALL values in "w" must be JSON-serializable (int, float, str — no tensors).
        Use random sampling — do not always return the same values.
        Vary the full range of each parameter, especially the ones that most
        affect algorithm behavior (load distribution shape, scale, expert count).

        Grammar for reference (use it to set valid ranges):
        ```json
        {grammar_json}
        ```

        Return ONLY a ```python ... ``` code block containing ONLY generate_workload.
    """)

    initial_path = app_dir / "initial_program.py"
    code = ""
    last_error = ""

    for attempt in range(MAX_RETRIES):
        prompt = code_prompt if attempt == 0 else textwrap.dedent(f"""
            The previous code failed with this error:
            {last_error}

            Original code:
            ```python
            {code}
            ```

            Fix the error. Return ONLY a corrected ```python ... ``` code block.
        """)

        response = client.messages.create(
            model=MODEL, max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        code = extract_code_block(response.content[0].text)
        if not code:
            last_error = "no code block returned"
            continue

        ok, last_error = _validate_generate_workload(code, app_dir, initial_path)
        if ok:
            print(f"[agent1] validation passed on attempt {attempt + 1}")
            break
        print(f"[agent1] attempt {attempt + 1} failed: {last_error}", file=sys.stderr)
    else:
        print(f"[agent1] WARNING: all {MAX_RETRIES} attempts failed. Using best-effort code.", file=sys.stderr)

    # Write outputs
    results_dir.mkdir(parents=True, exist_ok=True)
    grammar_path = results_dir / "grammar.json"
    grammar_path.write_text(json.dumps(grammar, indent=2))

    generate_path = results_dir / "generate_workload.py"
    generate_path.write_text(code)

    print(f"[agent1] wrote {grammar_path} and {generate_path}")
    return {"grammar": grammar, "code": code}
