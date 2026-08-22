import json
import shutil
import subprocess
from pathlib import Path

import pytest

from wmfs_tool.cli import main
from wmfs_tool.generator import generate

ROOT = Path(__file__).parents[3]
INTERFACE = ROOT / "plugins/reference/interface.toml"


def test_generation_is_deterministic_and_check_detects_stale(tmp_path: Path) -> None:
    first = generate(INTERFACE, tmp_path)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert generate(INTERFACE, tmp_path, check=True) == first
    generate(INTERFACE, tmp_path)
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["interfaceFingerprint"] == f"sha256:{first}"
    assert manifest["operationCount"] == 6
    assert manifest["plugin"]["worker"] == "wmfs-reference-worker"
    assert manifest["operations"][5]["outputs"][0]["allocation"] == "dynamic"

    (tmp_path / "manifest.json").write_text("stale")
    with pytest.raises(RuntimeError, match="manifest.json"):
        generate(INTERFACE, tmp_path, check=True)


def test_cli_reports_validation_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid = tmp_path / "invalid.toml"
    invalid.write_text("format_version = 1\n")

    assert (
        main(["generate", "--interface", str(invalid), "--output", str(tmp_path)]) == 1
    )
    assert "missing required field" in capsys.readouterr().err


def test_generated_headers_and_stub_compile_as_cpp11(tmp_path: Path) -> None:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler is available")
    generate(INTERFACE, tmp_path)
    source = tmp_path / "compile.cpp"
    source.write_text(
        "#include <wmfs/reference_plugin.hpp>\n"
        'static_assert(WMFS_REFERENCE_ABI_VERSION == 1, "ABI");\n'
        "int main() { return wmfs::reference::matmul == 1 ? 0 : 1; }\n"
    )
    subprocess.run(
        [
            compiler,
            "-std=c++11",
            "-Werror",
            "-I",
            str(tmp_path / "include"),
            "-c",
            str(source),
            "-o",
            str(tmp_path / "compile.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            compiler,
            "-std=c++11",
            "-Werror",
            "-I",
            str(tmp_path / "include"),
            "-c",
            str(tmp_path / "src/reference_plugin_stub.cpp"),
            "-o",
            str(tmp_path / "stub.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
