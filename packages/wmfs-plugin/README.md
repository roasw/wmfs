# wmfs-plugin

`wmfs-plugin` is the worker-side Python SDK for wmfs plugins. It provides the
Cap'n Proto protocol schemas, shared-memory tensor mapping, and generic worker
bootstrap without depending on the main `wmfs` execution runtime.

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
