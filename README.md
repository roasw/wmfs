# wmfs

`wmfs` is a prototype scientific-computing runtime for transparently running
selected Python function calls in isolated worker processes. The first
milestone provides the local PyTorch backend that will serve as the behavioral
and performance baseline for isolated execution.

## Development

Enter the Nix development shell and run the tests:

```console
nix develop
pytest
```

## Usage

```python
import torch

from wmfs import add_scalar, matmul, runtime, svd

a = torch.randn(4, 4)
b = torch.randn(4, 4)

runtime.use_backend("local")
c = matmul(a, b)
u, s, vh = svd(c)
d = add_scalar(c, 1.0)
```

The public calls will remain unchanged when the isolated backend is added.

## Plugin Discovery

Plugin deployment metadata identifies a worker module and its Cap'n Proto
schema. Operation signatures are declared once in the schema and discovered
over RPC when the plugin is registered:

```python
from pathlib import Path

from wmfs import runtime

runtime.discover_plugins(Path("plugins"))
print(runtime.operation_names)
```

Tensor payload transport is not part of this milestone. Discovered operations
continue to execute through the explicitly selected backend.
