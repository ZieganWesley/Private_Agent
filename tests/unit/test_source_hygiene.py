"""Guards that production code never branches on the test runner (issue A3)."""

import ast
import pathlib
import re

SOURCE_ROOT = pathlib.Path("src")

TEST_RUNNER_MARKERS = (
    '"pytest" in sys.modules',
    "'pytest' in sys.modules",
    'sys.modules["pytest"]',
    "sys.modules['pytest']",
    "PYTEST_CURRENT_TEST",
    "sys_modules_safe",
)


def test_production_sources_never_branch_on_the_test_runner():
    offenders = [
        f"{path.relative_to(SOURCE_ROOT)}: {marker}"
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        for marker in TEST_RUNNER_MARKERS
        if marker in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def _dotted_name(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


class _BlockingCallCollector(ast.NodeVisitor):
    def __init__(self):
        self.function_stack: list[tuple[str, bool]] = []
        self.calls: list[tuple[str, str | None, bool, int]] = []

    def visit_FunctionDef(self, node):
        self.function_stack.append((node.name, isinstance(node, ast.AsyncFunctionDef)))
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        name = _dotted_name(node.func)
        if name in {"asyncio.run", "subprocess.run"}:
            function, is_async = (
                self.function_stack[-1] if self.function_stack else (None, False)
            )
            self.calls.append((name, function, is_async, node.lineno))
        self.generic_visit(node)


def _blocking_calls():
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        collector = _BlockingCallCollector()
        collector.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for name, function, is_async, line in collector.calls:
            yield path.relative_to(SOURCE_ROOT), name, function, is_async, line


def test_asyncio_run_is_confined_to_the_sync_cli_boundary():
    offenders = [
        f"{path}:{line} ({function})"
        for path, name, function, _is_async, line in _blocking_calls()
        if name == "asyncio.run"
        and not (str(path) == "agent/runtime.py" and function == "run_agent_cli")
    ]
    assert offenders == []


def test_subprocess_run_never_runs_inside_async_code():
    offenders = [
        f"{path}:{line} ({function})"
        for path, name, function, is_async, line in _blocking_calls()
        if name == "subprocess.run" and is_async
    ]
    assert offenders == []


def _handled_names(node):
    """Collect exception type names from an `except` clause node."""
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Tuple):
        names = set()
        for element in node.elts:
            names |= _handled_names(element)
        return names
    return set()


def _broad_except_handlers():
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if not _handled_names(node.type) & {"Exception", "BaseException"}:
                continue
            lineno = getattr(node.type, "lineno", node.lineno)
            line = lines[lineno - 1] if lineno - 1 < len(lines) else ""
            yield path.relative_to(SOURCE_ROOT), lineno, line


def test_every_blind_exception_handler_carries_a_written_justification():
    """Issue 5.3: a blind catch must be a documented boundary, not a default."""
    justified = re.compile(r"#\s*noqa:\s*BLE001\b(.*)$")
    offenders = []
    for path, lineno, line in _broad_except_handlers():
        match = justified.search(line)
        if match is None:
            offenders.append(f"{path}:{lineno} has no '# noqa: BLE001 - reason'")
            continue
        if not match.group(1).strip(" -:\t"):
            offenders.append(f"{path}:{lineno} needs a reason after the noqa")
    assert offenders == []


def test_every_top_level_source_module_is_shipped():
    """A top-level module missing from ``py-modules`` breaks the installed wheel."""
    import tomllib

    pyproject = pathlib.Path("pyproject.toml")
    declared = set(
        tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["setuptools"][
            "py-modules"
        ]
    )
    modules = {path.stem for path in SOURCE_ROOT.glob("*.py")}
    assert modules - declared == set()


def test_log_event_never_uses_a_string_level():
    """log_event passes ``level`` to logging, so it must be an int (Part I)."""
    offenders = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or _dotted_name(node.func) != "log_event":
                continue
            for keyword in node.keywords:
                if keyword.arg != "level":
                    continue
                if isinstance(keyword.value, ast.Constant) and isinstance(
                    keyword.value.value, str
                ):
                    offenders.append(
                        f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}: "
                        f"level={keyword.value.value!r}"
                    )
    assert offenders == []
