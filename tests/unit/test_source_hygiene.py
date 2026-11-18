"""Guards that production code never branches on the test runner (issue A3)."""

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
