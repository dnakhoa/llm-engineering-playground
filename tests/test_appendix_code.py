"""
Appendix code runs on current models, through the provider layer (ticket 14).

What a Reader copies out of an Appendix script, notebook or TypeScript example has
to work on the first run. That means four things, each checked here by reading the
code, never by running it (the Appendix calls live APIs):

1. No Appendix code file is still in the stale-reference lint baseline.
2. Python goes through `llm/` — `ask()` or `complete()` — not the old
   `shared/provider.py`, and not a vendor SDK's chat call, so the registry decides
   what each model is sent.
3. Every chat-model ID the code names is in the model registry.
4. No temperature is handed straight to a vendor API. Through `llm/` it is dropped
   for any model that rejects it; a direct call has no such guard.

The MCP pages and `agentic_workflows.ipynb` belong to ticket 15 and are left out.

Run: pytest tests/test_appendix_code.py -v
"""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import stale_lint  # noqa: E402
from llm import load_registry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
APPENDIX = REPO_ROOT / "appendix"

#: Refreshed by ticket 15 (MCP 2026-07-28 and agent frameworks), not here.
LATER_BATCH = ("appendix/mcp/", "appendix/agent-frameworks/agentic_workflows.ipynb")

CODE_SUFFIXES = (".py", ".ipynb", ".ts")

#: Files that call a vendor SDK directly, and why the provider layer cannot serve them.
#: They are still held to current model IDs and no temperature.
DIRECT_SDK_CALLS = {
    "appendix/multimodal/multimodal_example.py": "images and audio: the layer is text-only",
    "appendix/multimodal/multimodal_example.ipynb": "images and audio: the layer is text-only",
    "appendix/context-engineering/context_engineering.py": "explicit cache_control breakpoints",
    "appendix/deployment/deployment.ipynb": "token streaming: the layer returns whole responses",
}

#: Chat calls on the vendor SDKs, and LangChain's chat-model wrappers.
_VENDOR_CHAT_CALL = re.compile(
    r"\.chat\.completions\.create\s*\(|\.messages\.create\s*\(|\.responses\.create\s*\("
    r"|\.responses\.stream\s*\(|\.messages\.stream\s*\(|\.generate_content\s*\("
    r"|\bChat(?:OpenAI|Anthropic|GoogleGenerativeAI)\s*\("
)

#: A quoted model ID from a provider the registry covers.
_MODEL_ID = re.compile(r"""["'`]((?:gpt|claude|gemini|grok|deepseek)-[A-Za-z0-9.\-]*)["'`]""")

#: OpenAI models the registry does not list because they are not chat models.
NON_CHAT_MODELS = frozenset({"gpt-image-1"})

REGISTRY_IDS = frozenset(load_registry().ids())


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _in_scope(relative: str) -> bool:
    return not any(relative == p or relative.startswith(p) for p in LATER_BATCH)


def appendix_code() -> list[Path]:
    return sorted(
        path
        for path in APPENDIX.rglob("*")
        if path.suffix in CODE_SUFFIXES
        and ".ipynb_checkpoints" not in path.parts
        and _in_scope(_relative(path))
    )


def code_of(path: Path) -> list[tuple[str, str]]:
    """(where, source) for a script, or for each code cell of a notebook."""
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return [("", text)]
    sources = []
    for index, cell in enumerate(json.loads(text)["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        # %magic and !shell lines are IPython, not Python.
        lines = ("" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines())
        sources.append((f"cell {index} ", "\n".join(lines)))
    return sources


CODE = appendix_code()
PYTHON = [p for p in CODE if p.suffix in (".py", ".ipynb")]
IDS = [_relative(p) for p in CODE]
PYTHON_IDS = [_relative(p) for p in PYTHON]


def _without_strings_and_comments(source: str) -> str:
    """Python source with comments and string literals blanked, line numbers kept.

    A docstring that says "stands in for client.messages.create(...)" is not a call.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    lines = source.splitlines(keepends=True)
    for token in reversed(tokens):
        if token.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (start_row, start_col), (end_row, end_col) = token.start, token.end
        for row in range(start_row, end_row + 1):
            line = lines[row - 1]
            begin = start_col if row == start_row else 0
            end = end_col if row == end_row else len(line.rstrip("\n"))
            lines[row - 1] = line[:begin] + " " * (end - begin) + line[end:]
    return "".join(lines)


def _line(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def test_the_scan_sees_the_appendix_code():
    # An empty scan would pass everything below on nothing.
    assert "appendix/foundations/llm_foundations.py" in IDS
    assert "appendix/rag/rag_systems.ipynb" in IDS
    assert "appendix/typescript/agent.ts" in IDS
    assert not any(label.startswith("appendix/mcp/") for label in IDS)


def test_no_appendix_code_file_is_in_the_lint_baseline():
    baselined = sorted(label for label in stale_lint.load_baseline() if label in IDS)

    assert baselined == [], "fix these and take them out of tests/stale_lint_baseline.txt"


@pytest.mark.parametrize("path", PYTHON, ids=PYTHON_IDS)
def test_python_does_not_use_the_old_provider(path):
    found = []
    for where, source in code_of(path):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            if any(name in ("provider", "shared.provider") for name in names):
                found.append(f"{where}line {node.lineno}")

    assert found == [], f"{_relative(path)} imports shared/provider.py; use `from llm import ask`"


@pytest.mark.parametrize("path", PYTHON, ids=PYTHON_IDS)
def test_python_chat_calls_go_through_the_provider_layer(path):
    if _relative(path) in DIRECT_SDK_CALLS:
        pytest.skip(DIRECT_SDK_CALLS[_relative(path)])
    found = [
        f"{where}line {_line(source, m.start())}: {m.group(0).strip()}"
        for where, source in code_of(path)
        for m in _VENDOR_CHAT_CALL.finditer(_without_strings_and_comments(source))
    ]

    assert found == [], f"{_relative(path)} calls a vendor chat API directly; use llm.ask or llm.complete"


@pytest.mark.parametrize("path", CODE, ids=IDS)
def test_chat_model_ids_are_in_the_registry(path):
    unknown = [
        f"{where}line {_line(source, m.start())}: {m.group(1)}"
        for where, source in code_of(path)
        for m in _MODEL_ID.finditer(source)
        if m.group(1) not in REGISTRY_IDS | NON_CHAT_MODELS
    ]

    assert unknown == [], f"{_relative(path)} names models missing from llm/models.json"


def _temperature_sent_directly(source: str) -> list[int]:
    """Lines where a temperature reaches a vendor call without going through llm/."""
    tree = ast.parse(source)
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(k.arg == "temperature" for k in node.keywords):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in {"ask", "CallOptions", "generate", "GenerationConfig"}:
            # llm/ drops it where rejected; `generate` is a local Hugging Face model.
            continue
        lines.append(node.lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == "temperature":
                    lines.append(node.lineno)
    return sorted(lines)


@pytest.mark.parametrize("path", CODE, ids=IDS)
def test_no_temperature_goes_straight_to_a_vendor_api(path):
    found = []
    for where, source in code_of(path):
        if path.suffix == ".ts":
            # TypeScript has no provider layer to drop it, so it never sends one.
            found += [f"line {_line(source, m.start())}" for m in re.finditer(r"\btemperature\s*:", source)]
            continue
        try:
            found += [f"{where}line {n}" for n in _temperature_sent_directly(source)]
        except SyntaxError:
            continue

    assert found == [], (
        f"{_relative(path)} sends temperature itself; pass it to llm.ask/complete, "
        "which drops it for models that reject it"
    )


def test_a_call_named_in_a_docstring_is_not_a_call():
    source = '"""Stands in for client.messages.create(...)."""\nx = 1  # not .responses.create(\n'

    assert _VENDOR_CHAT_CALL.search(_without_strings_and_comments(source)) is None
    assert _VENDOR_CHAT_CALL.search("client.messages.create(model=m)")


def test_the_temperature_check_catches_a_direct_call():
    source = 'client.chat.completions.create(model=m, messages=[], temperature=0.7)\n'

    assert _temperature_sent_directly(source) == [1]


def test_the_temperature_check_allows_the_provider_layer():
    source = 'ask("hi", temperature=0.7)\nCallOptions(temperature=0.2)\n'

    assert _temperature_sent_directly(source) == []
