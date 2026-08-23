import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "tests/compatibility/gcc48_compile.sh"


def test_gcc48_script_syntax_inputs_and_scope() -> None:
    bash = shutil.which("bash")
    assert bash is not None

    subprocess.run([bash, "-n", str(SCRIPT)], check=True)
    subprocess.run([bash, str(SCRIPT), "--check-inputs"], check=True)

    source = SCRIPT.read_text(encoding="utf-8")
    assert "cc=/usr/bin/gcc" in source
    assert "cxx=/usr/bin/g++" in source
    assert '[[ "$version" != 4.8.5 ]]' in source
    assert "-std=c99" in source
    assert "-std=c11" in source
    assert "-std=c++11 -Wall -Wextra -Werror" in source
    assert "plugins/reference/generated" in source
    assert "tests/fixtures/mode_neutral/generated" in source
    assert "tests/integration/fixtures/generated-v1" in source
    assert "plugins/reference/worker" not in source
    assert "plugins/reference/src" not in source
