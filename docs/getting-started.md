# Getting Started

The public API does not expose RPC requests, workers, or shared-memory handles.
The same functions are used for every backend.

```python
from pathlib import Path

import torch

from wmfs import add_scalar, matmul, runtime

runtime.discover_plugins(Path("plugins"))
runtime.use_backend("isolated")

a = torch.randn(4, 3, requires_grad=True)
b = torch.randn(3, 2, requires_grad=True)
c = add_scalar(matmul(a, b), 1.0)
c.square().sum().backward()

print(c)
print(a.grad, b.grad)
runtime.close()
```

Use `local` for direct PyTorch execution, `bundled` for the in-process reference
plugin when it was compiled, and `isolated` for process isolation. Plugin
discovery starts and validates persistent workers eagerly, so the first
operation does not start a second process.

Isolated autograd uses plugin-advertised first-order VJPs. The main process
retains PyTorch graph scheduling; the plugin implements the mathematical VJP.
Higher-order isolated gradients and isolated SVD gradients are currently
rejected explicitly.
