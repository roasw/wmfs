from pathlib import Path

import pytest

from wmfs.transport.deadlines import TransportDeadlines


@pytest.fixture
def short_transport_deadlines() -> TransportDeadlines:
    # Importing the SDK also imports torch; keep startup tolerant while making
    # every transport failure complete far below the production defaults.
    return TransportDeadlines(2.0, 0.1, 0.08, 1.0, 0.2)


TEST_LAYERS = {
    "contract": {"test_backend_contract.py"},
    "integration": {
        "test_dependency_boundary.py",
        "test_environment.py",
        "test_generated_v1_compatibility.py",
        "test_invocation.py",
        "test_isolated.py",
        "test_logging.py",
        "test_output_metadata.py",
        "test_plugins.py",
        "test_tensor_transport.py",
    },
    "package": {
        "test_benchmark.py",
        "test_bundled.py",
        "test_documentation_versions.py",
        "test_version.py",
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
