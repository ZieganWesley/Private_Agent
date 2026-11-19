"""Guards that production code never branches on the test runner (issue A3)."""

import ast
import inspect
import pathlib

import private_agent

SOURCE_ROOT = pathlib.Path(inspect.getfile(private_agent)).parent

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
