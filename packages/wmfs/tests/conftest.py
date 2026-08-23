from pathlib import Path

import pytest

UNIT_TESTS = {
    "test_api.py",
    "test_backend_lifecycle.py",
    "test_configuration.py",
    "test_control.py",
    "test_dynamic_api.py",
    "test_logging.py",
    "test_memory.py",
    "test_registry.py",
    "test_runtime_lifecycle.py",
}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    del config
    tests_directory = Path(__file__).parent
    test_files = {path.name for path in tests_directory.glob("test_*.py")}
    if test_files != UNIT_TESTS:
        raise pytest.UsageError(
            "stale runtime unit-test classification: "
            f"missing={sorted(UNIT_TESTS - test_files)}, "
            f"unclassified={sorted(test_files - UNIT_TESTS)}"
        )
    for item in items:
        if item.path.parent == tests_directory:
            item.add_marker("unit")
