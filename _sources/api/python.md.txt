# Python API Reference

Python documentation is generated from source docstrings with Sphinx autodoc.
Public docstrings use Google-style parameter and return sections.

## User API

Tensor constructors are static API members:

```{eval-rst}
.. automodule:: wmfs.api
   :members:
```

Plugin operations are dynamic attributes of `wmfs`. They become available only
after `Runtime.discover_plugins` publishes worker metadata or after an
in-process backend is explicitly selected. The reference plugin publishes:

```python
wmfs.matmul(a, b, *, out=None)
wmfs.svd(a, full_matrices=True, *, out=None)
wmfs.add_scalar(a, value, *, out=None)
```

Other plugins publish their own schema-declared operation names; internal
operations are never exposed as module attributes.

## Runtime

```{eval-rst}
.. autoclass:: wmfs.runtime.Runtime
   :members:

.. autoclass:: wmfs.transport.deadlines.TransportDeadlines
   :members:
```

## Plugin SDK

```{eval-rst}
.. autoclass:: wmfs_plugin.invocation.InvocationContext
   :members:

.. autofunction:: wmfs_plugin.worker.worker_main
```

```{eval-rst}
.. automodule:: wmfs_plugin.metadata
   :members: PluginMetadata, OperationMetadata, TensorParameter, ScalarParameter, OutputPlan, VjpMetadata, metadata_from_reader, validate_plugin_metadata, metadata_fingerprint
```
