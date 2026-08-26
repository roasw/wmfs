# Getting Started

The public API does not expose requests, rings, workers, futures, or shared-memory
handles. Calls are synchronous even though the isolated runtime supports
multiple asynchronous submissions internally. The same functions are used for
every backend.

Application environments install `wmfs`. Python worker environments install
`wmfs-plugin`; C++ workers do not. Plugin build and CI environments install
`wmfs-tool` to generate committed deployment artifacts. See
{doc}`package-roles` for the complete dependency and workflow matrix.
For the complete visual path from API call to worker and shared result, follow
the {doc}`architecture-overview`.

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

To inspect and configure a plugin without importing or launching it, separate
manifest loading from initialization:

```python
from wmfs import LoggingOptions

wmfs.runtime.load_plugins(Path("plugins"))
schema = wmfs.list_configurable("reference")
wmfs.runtime.validate_config("reference", {"threads": 4})
wmfs.runtime.configure_plugin(
    "reference",
    {"threads": 4},
    logging=LoggingOptions(mode="centralized", level=20),
)
wmfs.runtime.discover_plugins(Path("plugins"))
wmfs.runtime.use_backend("isolated")
```

Configuration is canonicalized once and immutable after initialization. Bundled
initialization occurs when that backend is selected; isolated
initialization occurs during eager discovery. The default disabled logger
creates no logging transport or queue. See {doc}`ring-protocol`.

Use `bundled` for in-process Python or native plugins and `isolated` for process
isolation. Plugin
discovery starts and validates persistent workers eagerly, so the first
operation does not start a second process.

Operation names are dynamic module attributes. Before plugin discovery or an
explicit `bundled` backend selection, names such as `wmfs.matmul` do not
exist. Import `wmfs` first, configure the runtime, and then access its published
operations.

`empty`, `zeros`, `ones`, and `randn` return ordinary Torch tensors. With the
isolated backend selected, their storage is allocated directly in shared memory
and avoids an ingress copy on the first worker invocation. With `bundled`, they
delegate to native Torch constructors.

Isolated autograd uses plugin-advertised first-order VJPs. The main process
retains PyTorch graph scheduling; the plugin implements the mathematical VJP.
Higher-order isolated gradients and isolated SVD gradients are currently
rejected explicitly.
