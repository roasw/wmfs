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
