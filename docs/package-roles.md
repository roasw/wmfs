# Package Roles

WMFS publishes three Python distributions with deliberately different users
and lifetimes. They are not a stack that every environment should install.

| Distribution  | Installed by                     | Used when                       | Purpose                                                                                                        |
| ------------- | -------------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `wmfs`        | Application/runtime environment  | Runtime                         | Public Python API, plugin discovery, worker lifecycle, rings, shared buffers, and tensor results               |
| `wmfs-plugin` | Python plugin worker environment | Runtime, inside the worker only | Stable Python worker bootstrap, invocation context, FD receiver, mapped tensors, and worker-side ring endpoint |
| `wmfs-tool`   | Plugin build and CI environment  | Build time only                 | Compiles `interface.toml` into manifests and C++/Python plugin-facing artifacts                                |

## `wmfs`: Application Runtime

Applications install `wmfs` and call ordinary operations:

```console
python -m pip install wmfs
```

```python
from pathlib import Path

import wmfs

wmfs.runtime.discover_plugins(Path("plugins"))
wmfs.runtime.use_backend("isolated")
result = wmfs.matmul(left, right)
```

The distribution owns the application-side protocol model, startup schemas,
ring producer/completion dispatcher, shared-memory allocator, FD sender, and
worker process lifecycle. It reads generated plugin manifests but does not
import plugin implementations.

`wmfs` does not depend on `wmfs-plugin` or `wmfs-tool`. A local, bundled, or C++
isolated deployment therefore does not install the Python worker SDK or the
interface compiler.

## `wmfs-plugin`: Python Worker SDK

Install `wmfs-plugin` only in an environment that executes a Python plugin
worker:

```console
python -m pip install wmfs-plugin
```

Generated Python adapters import its stable worker-facing types. Plugin code
uses `InvocationContext` and `worker_main` rather than importing the application
runtime:

```python
from wmfs_plugin import InvocationContext, worker_main


def scale(context: InvocationContext) -> None:
    context.output("result").copy_(
        context.input("value") * context.scalar("factor")
    )


worker_main({"scale": scale})
```

The SDK owns the Python worker bootstrap, mapped-buffer receiver, worker-side
ring endpoint, and compatible worker-side protocol definitions. It does not
depend on `wmfs` or `wmfs-tool`.

Do not add `wmfs-plugin` to an application's dependencies merely because the
application invokes plugins. The worker package declares it when, and only
when, that worker is implemented in Python. C++ workers do not use it.

## `wmfs-tool`: Interface Compiler

Plugin projects use `wmfs-tool` while building or checking generated artifacts:

```console
python -m pip install wmfs-tool
wmfs-tool generate --interface interface.toml --output generated
wmfs-tool generate --interface interface.toml --output generated --check
```

The TOML interface is the handwritten source of truth. Generated outputs
include:

- `manifest.json`, consumed by `wmfs` before worker launch;
- C and C++11 ABI headers and dispatch stubs;
- Python metadata, type stubs, and implementation adapters.

Generated artifacts are committed to or packaged with the plugin. Neither the
application runtime nor the deployed worker runs `wmfs-tool`, so it is not a
runtime dependency. The compiler itself has no dependency on `wmfs`,
`wmfs-plugin`, PyTorch, or runtime transport libraries.

## Plugin Workflows

### Python Plugin

1. Declare operations in `interface.toml`.
1. Install and run `wmfs-tool` in the build environment.
1. Implement the generated Python interface using `wmfs-plugin` types.
1. Package the generated manifest with the worker distribution.
1. Declare `wmfs-plugin` and the numerical library as worker runtime
   dependencies; do not declare `wmfs` or `wmfs-tool`.

The application and Python worker may use different Python, PyTorch, libc, and
dependency environments. Their protocol copies are tested for wire parity but
are packaged independently so neither environment imports the other package.

### C++ Plugin

1. Declare operations in `interface.toml`.
1. Run `wmfs-tool` during the build.
1. Implement the generated C++11-compatible interface.
1. Package the generated manifest and worker executable together.

The deployed C++ worker requires neither `wmfs-plugin` nor `wmfs-tool`. It
communicates with `wmfs` only through the generated startup contract,
command/completion rings, the FD-control channel, and shared tensor mappings.

## Dependency Direction

```text
application
    -> wmfs

Python worker distribution
    -> wmfs-plugin

plugin build / CI
    -> wmfs-tool

wmfs-tool
    -> generated manifest + C++ interface + Python interface
```

There is intentionally no dependency edge from `wmfs` to `wmfs-plugin`, from
`wmfs-plugin` to `wmfs`, or from either runtime package to `wmfs-tool`.
