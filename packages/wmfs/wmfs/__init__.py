from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from wmfs.api import add_scalar, matmul, svd
from wmfs.runtime import runtime

__all__ = ["add_scalar", "matmul", "runtime", "svd"]
