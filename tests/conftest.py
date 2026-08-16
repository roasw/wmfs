from pathlib import Path

import pytest

TEST_LAYERS = {
    "unit": {
        "test_api.py",
        "test_backend_lifecycle.py",
        "test_fd_broker.py",
        "test_invocation.py",
        "test_memory.py",
        "test_output_metadata.py",
        "test_registry.py",
        "test_runtime_lifecycle.py",
    },
    "contract": {"test_backend_contract.py"},
    "integration": {
        "test_environment.py",
        "test_isolated.py",
        "test_plugins.py",
        "test_tensor_transport.py",
    },
    "package": {
        "test_benchmark.py",
        "test_bundled.py",
        "test_codegen.py",
    },
    "native": {"test_native_session.py"},
}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    tests_directory = Path(__file__).parent
    test_files = {path.name for path in tests_directory.glob("test_*.py")}
    classified_files = set().union(*TEST_LAYERS.values())
    empty_layers = [name for name, files in TEST_LAYERS.items() if not files]
    duplicate_files = sorted(
        filename
        for filename in classified_files
        if sum(filename in files for files in TEST_LAYERS.values()) != 1
    )
    missing_files = classified_files - test_files
    unclassified_files = test_files - classified_files
    if empty_layers or duplicate_files or missing_files or unclassified_files:
        raise pytest.UsageError(
            "stale test layer classification: "
            f"empty={empty_layers}, duplicates={duplicate_files}, "
            f"missing={sorted(missing_files)}, "
            f"unclassified={sorted(unclassified_files)}"
        )

    layers_by_file = {
        filename: layer
        for layer, filenames in TEST_LAYERS.items()
        for filename in filenames
    }
    for item in items:
        try:
            filename = item.path.relative_to(tests_directory).parts[0]
        except ValueError:
            continue
        if filename.startswith("test_"):
            item.add_marker(layers_by_file[filename])
