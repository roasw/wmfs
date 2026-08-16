# Python API Reference

Python documentation is generated from source docstrings with Sphinx autodoc.
Public docstrings use Google-style parameter and return sections.

## User API

```{eval-rst}
.. automodule:: wmfs.api
   :members:
```

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
