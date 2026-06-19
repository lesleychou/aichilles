"""Shared helpers: template rendering, code extraction, safe exec."""
import re
from pathlib import Path
from typing import Any


MODEL = "claude-sonnet-4-6"


def render_template(template_path: Path, variables: dict[str, str]) -> str:
    """Replace {{key}} placeholders in template file with values."""
    text = template_path.read_text()
    for key, value in variables.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text


def extract_code_block(text: str) -> str:
    """Extract first ```python ... ``` block from LLM response."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else ""


def safe_exec(code: str, fn_name: str) -> tuple[Any, str | None]:
    """
    exec() code and call fn_name().
    Returns (result, None) on success or (None, error_message) on failure.
    """
    try:
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102
        fn = ns.get(fn_name)
        if not callable(fn):
            return None, f"{fn_name}() not defined in code"
        return fn(), None
    except Exception as exc:
        return None, str(exc)
