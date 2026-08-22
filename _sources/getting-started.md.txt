# Getting Started

The public API does not expose RPC requests, workers, or shared-memory handles.
The same functions are used for every backend.

```python
from pathlib import Path

import wmfs

wmfs.runtime.discover_plugins(Path("plugins"))
wmfs.runtime.use_backend("isolated")

a = wmfs.randn(4, 3, requires_grad=True)
b = wmfs.randn(3, 2, requires_grad=True)
c = wmfs.add_scalar(wmfs.matmul(a, b), 1.0)
c.square().sum().backward()

print(c)
print(a.grad, b.grad)
wmfs.runtime.close()
```

Use `local` for direct PyTorch execution, `bundled` for the in-process reference
plugin when it was compiled, and `isolated` for process isolation. Plugin
discovery starts and validates persistent workers eagerly, so the first
operation does not start a second process.

Operation names are dynamic module attributes. Before plugin discovery or an
explicit `local`/`bundled` backend selection, names such as `wmfs.matmul` do not
exist. Import `wmfs` first, configure the runtime, and then access its published
operations.

`empty`, `zeros`, `ones`, and `randn` return ordinary Torch tensors. With the
isolated backend selected, their storage is allocated directly in shared memory
and avoids an ingress copy on the first worker invocation. With `local` or
`bundled`, they delegate to the corresponding native Torch constructors.

Isolated autograd uses plugin-advertised first-order VJPs. The main process
retains PyTorch graph scheduling; the plugin implements the mathematical VJP.
Higher-order isolated gradients and isolated SVD gradients are currently
rejected explicitly.
