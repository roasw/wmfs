import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from wmfs.plugins import load_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "generated-v1"
FINGERPRINT = "b0f2005fae597c883fc5caf85c3b78f88996b0141b5c7a71bdd9d7163efc3123"


def _deployed_manifest(
    tmp_path: Path, document: dict[str, object] | None = None
) -> Path:
    plugin_root = tmp_path / "reference"
    generated = plugin_root / "generated"
    schema = plugin_root / "schemas" / "wmfs-reference" / "reference.capnp"
    generated.mkdir(parents=True)
    schema.parent.mkdir(parents=True)
    schema.write_text("# temporary compatibility-test schema\n", encoding="utf-8")
    destination = generated / "manifest.json"
    if document is None:
        shutil.copyfile(FIXTURE / "manifest.json", destination)
    else:
        destination.write_text(json.dumps(document), encoding="utf-8")
    return destination


def _fixture_manifest() -> dict[str, object]:
    return json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))


def test_runtime_decodes_frozen_v1_manifest_without_regeneration(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(_deployed_manifest(tmp_path))

    assert manifest.name == "reference"
    assert manifest.interface == "ReferencePlugin"
    assert manifest.metadata.fingerprint == 0x549FB18B7A4B6C75
    assert tuple(operation.name for operation in manifest.metadata.operations) == (
        "matmul",
        "svd",
        "add_scalar",
        "matmul_vjp",
        "add_scalar_vjp",
        "nonzero",
    )
    assert manifest.metadata.operations[0].operation_id == 1
    assert manifest.metadata.operations[1].scalar_parameters[0].default is True
    assert manifest.metadata.operations[-1].output_plans[0].known is None
    assert manifest.schema_path.is_file()
    assert manifest.root == tmp_path / "reference"
    assert not (FIXTURE / "interface.toml").exists()


def test_frozen_v1_python_metadata_imports_with_python_311_grammar() -> None:
    path = FIXTURE / "python" / "wmfs_reference" / "interface.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path), feature_version=(3, 11))
    stub = FIXTURE / "python" / "wmfs_reference" / "interface.pyi"
    ast.parse(
        stub.read_text(encoding="utf-8"),
        filename=str(stub),
        feature_version=(3, 11),
    )
    spec = importlib.util.spec_from_file_location("frozen_generated_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        sys.modules.pop(spec.name, None)

    assert isinstance(module, ModuleType)
    assert module.INTERFACE_FINGERPRINT == f"sha256:{FINGERPRINT}"
    assert tuple(module.OPERATIONS_BY_NAME) == (
        "matmul",
        "svd",
        "add_scalar",
        "matmul_vjp",
        "add_scalar_vjp",
        "nonzero",
    )
    assert not tuple((path.parent / "__pycache__").glob("*"))


def test_frozen_v1_headers_and_stub_compile_as_cpp11(tmp_path: Path) -> None:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler is available")
    probe = tmp_path / "probe.cpp"
    probe.write_text(
        "#include <wmfs/reference_plugin.hpp>\n"
        'static_assert(WMFS_REFERENCE_ABI_VERSION == 1, "ABI");\n'
        "int main() { return wmfs::reference::nonzero == 6 ? 0 : 1; }\n",
        encoding="utf-8",
    )
    for source in (probe, FIXTURE / "src" / "reference_plugin_stub.cpp"):
        subprocess.run(
            [
                compiler,
                "-std=c++11",
                "-Werror",
                "-I",
                str(FIXTURE / "include"),
                "-c",
                str(source),
                "-o",
                str(tmp_path / f"{source.stem}.o"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("formatVersion", 3, "runtime supports 1 and 2"),
        ("abiVersion", 2, "Manifest abiVersion is 2, but runtime requires 1"),
        (
            "generator",
            "wmfs-tool/1.1",
            "Manifest generator is 'wmfs-tool/1.1', but runtime requires 'wmfs-tool/1'",
        ),
    ],
)
def test_runtime_rejects_incompatible_v1_versions_clearly(
    tmp_path: Path, field: str, value: object, diagnostic: str
) -> None:
    document = _fixture_manifest()
    document[field] = value

    with pytest.raises(ValueError, match=diagnostic):
        load_manifest(_deployed_manifest(tmp_path, document))


def test_v1_closed_format_rejects_unknown_additive_features(tmp_path: Path) -> None:
    document = _fixture_manifest()
    document["features"] = ["future-optional-feature"]

    with pytest.raises(
        ValueError,
        match=r"manifest fields do not match.*unknown=\['features'\]",
    ):
        load_manifest(_deployed_manifest(tmp_path, document))
