"""WMFS plugin interface compiler."""

from wmfs_tool.generator import generate
from wmfs_tool.parser import InterfaceError, load_interface

__all__ = ["InterfaceError", "generate", "load_interface"]
