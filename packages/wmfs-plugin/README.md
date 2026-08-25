# wmfs-plugin

`wmfs-plugin` is the standalone Python SDK for bundled and isolated plugins. It
does not depend on or import the main `wmfs` execution runtime. Every Python
plugin uses the same binding facade in process or in a worker.

This package is not required by the main `wmfs` runtime or C++ plugins. Python
plugin distributions declare it directly. Interface generation belongs to
`wmfs-tool` and happens before deployment.

For v0.1, the invocation context, shared-memory tensor mapping, and generic
worker bootstrap are intentionally Torch-specific. The wire schemas and
`wmfs_plugin.metadata` contract remain Torch-independent, so control-plane code
can inspect and validate operation metadata without importing Torch. These
layers remain in one distribution; the current boundary does not warrant a
premature package split.

The SDK does not provide code generation. `wmfs-tool` is the sole generator;
its generated Python implementation adapters target these stable worker types.

`PluginBinding` carries direct operations and generated worker handlers for the
same catalog. Bundled execution calls the direct view; isolated workers consume
the invocation-context view:

```python
from wmfs_plugin import InvocationContext, PluginBinding, worker_main


def add(context: InvocationContext) -> None:
    context.output("result").copy_(context.input("left") + context.input("right"))


plugin = PluginBinding({"add": add}, {"add": lambda left, right: left + right})
worker_main(plugin)
```

The context exposes the operation metadata, invocation ID, and validated input,
output, and scalar tuples. `input()`, `output()`, and `scalar()` accept either a
metadata name or positional index. It intentionally does not expose mapped
buffer caches, RPC objects, or main-runtime internals.

Import Torch-independent protocol APIs from their modules:

```python
from wmfs_plugin.metadata import OperationMetadata, validate_operation_metadata
from wmfs_plugin.schema import PROTOCOL_VERSION, schema_root
```

The root package exports `PluginBinding`, `InvocationContext`,
`OperationHandler`, and `worker_main` lazily, so importing a metadata or schema
submodule does not initialize the Torch worker layer.
