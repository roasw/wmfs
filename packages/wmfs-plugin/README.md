# wmfs-plugin

`wmfs-plugin` is the standalone worker-side Python SDK for wmfs plugins. It does
not depend on or import the main `wmfs` execution runtime. Plugins can therefore
install and deploy the SDK with their worker environment rather than the
application environment.

For v0.1, the invocation context, shared-memory tensor mapping, and generic
worker bootstrap are intentionally Torch-specific. The wire schemas and
`wmfs_plugin.metadata` contract remain Torch-independent, so control-plane code
can inspect and validate operation metadata without importing Torch. These
layers remain in one distribution; the current boundary does not warrant a
premature package split.

Operation handlers accept one `InvocationContext` and return `None`:

```python
from wmfs_plugin import InvocationContext, worker_main


def add(context: InvocationContext) -> None:
    context.output("result").copy_(context.input("left") + context.input("right"))


worker_main({"add": add})
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

The root package continues to export the worker-facing `InvocationContext`,
`OperationHandler`, and `worker_main` API lazily, so importing a metadata or
schema submodule does not initialize the Torch worker layer.
