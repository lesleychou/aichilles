# aichilles_risk_discovery/schema_infer.py
"""
Stage 0: infer Grammar_config + Grammar_workload from evaluator source via Claude API.

infer_schema(app_dir, client) -> dict
  Returns {"grammar_config": [...], "grammar_workload": [...], "constraints": [...], "notes": "..."}
"""
import json
import re
import sys
from pathlib import Path

import anthropic

from utils import render_template, MODEL
from param_extractor import extract_params

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_DATA_EXTENSIONS = (".json", ".csv", ".tsv")


def _load_data_file_metadata(app_dir: Path) -> dict:
    """
    Load metadata (shape, keys, columns) from external data files.
    Does not load full file content.
    """
    metadata = {}
    for path in app_dir.iterdir():
        if path.suffix not in _DATA_EXTENSIONS:
            continue
        try:
            if path.suffix == ".json":
                import json as _json
                data = _json.loads(path.read_text())
                if isinstance(data, dict):
                    metadata[path.name] = {"type": "dict", "keys": list(data.keys())[:20]}
                elif isinstance(data, list):
                    metadata[path.name] = {"type": "list", "length": len(data),
                                           "sample_keys": list(data[0].keys())[:10]
                                           if data and isinstance(data[0], dict) else []}
            elif path.suffix in (".csv", ".tsv"):
                sep = "\t" if path.suffix == ".tsv" else ","
                first_line = path.open().readline().strip()
                cols = first_line.split(sep)
                metadata[path.name] = {"type": "csv", "columns": cols[:20]}
        except Exception:
            pass
    return metadata


def infer_schema(app_dir: Path, client: anthropic.Anthropic) -> dict:
    evaluator_path = app_dir / "evaluator.py"
    initial_path = app_dir / "initial_program.py"
    rw_path = app_dir / "run_workload.py"
    if not evaluator_path.exists():
        return {"grammar_config": [], "grammar_workload": [], "constraints": [], "notes": "evaluator.py not found"}

    evaluator_source = evaluator_path.read_text()
    initial_source = initial_path.read_text() if initial_path.exists() else ""
    # run_workload.py is the fixed per-app harness that defines the workload dict
    # schema (key names + types). Grammar parameter names must match exactly so
    # Agent 2's mutation LLM generates workloads run_workload.py can consume.
    run_workload_source = rw_path.read_text() if rw_path.exists() else ""

    try:
        extracted = extract_params(app_dir)
        extracted_json = json.dumps(extracted, indent=2)
    except Exception as exc:
        print(f"[schema_infer] param extraction failed: {exc}", file=sys.stderr)
        extracted_json = '{"extracted": [], "methods_used": []}'

    data_metadata = _load_data_file_metadata(app_dir)
    data_metadata_json = json.dumps(data_metadata, indent=2) if data_metadata else "{}"

    template_path = _TEMPLATES_DIR / "agent1_schema.txt"
    prompt = render_template(template_path, {
        "extracted_params_json": extracted_json,
        "data_file_metadata_json": data_metadata_json,
        "evaluator_source": evaluator_source,
        "initial_program_source": initial_source,
        "run_workload_source": run_workload_source,
        "knowledge_context": "(none)",
    })

    try:
        response = client.messages.create(
            model=MODEL, max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        json_match = re.search(r"```json\s*\n(.*?)```", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        return json.loads(raw)
    except Exception as exc:
        print(f"[schema_infer] LLM call failed: {exc}", file=sys.stderr)
        return {"grammar_config": [], "grammar_workload": [], "constraints": [], "notes": f"inference failed: {exc}"}
