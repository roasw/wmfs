import argparse
import sys
from pathlib import Path

from wmfs_tool.generator import generate
from wmfs_tool.parser import InterfaceError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wmfs-tool")
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser(
        "generate", help="generate plugin ABI artifacts from an interface"
    )
    generate_parser.add_argument("--interface", required=True, type=Path)
    generate_parser.add_argument("--output", required=True, type=Path)
    generate_parser.add_argument(
        "--check", action="store_true", help="fail instead of updating stale artifacts"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        fingerprint = generate(
            arguments.interface, arguments.output, check=arguments.check
        )
    except (InterfaceError, RuntimeError) as error:
        print(f"wmfs-tool: error: {error}", file=sys.stderr)
        return 1
    action = "checked" if arguments.check else "generated"
    print(f"{action} {arguments.output} (sha256:{fingerprint})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
