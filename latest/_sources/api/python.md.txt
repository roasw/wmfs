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

Operation calls synchronously return ordinary Torch tensors. The isolated
backend uses asynchronous command/completion rings internally, but the current
public API does not expose command records or mandatory futures. `ping()` and
`ring_ping()` are session-level benchmark internals, not user API.

## Plugin Author Flow

Write a versioned `interface.toml`, run `wmfs-tool generate`, and commit the
manifest, C/C++11 ABI files, and Python metadata/stubs. Package and runtime builds
consume those committed files. CI runs the same command with `--check`; plugins
do not depend on `wmfs-tool` at runtime. Python worker distributions depend on
`wmfs-plugin`; C++ workers do not.

## Runtime

```{eval-rst}
.. autoclass:: wmfs.runtime.Runtime
   :members:

.. autoclass:: wmfs.transport.deadlines.TransportDeadlines
   :members:
```

## Plugin SDK

`wmfs-plugin` is documented and tested as an independent Python-worker SDK.
Core runtime documentation deliberately does not import it: runtime and worker
SDK packages carry compatible protocol definitions but may be installed in
different environments. See `packages/wmfs-plugin/README.md` and generated
Python interface stubs for its worker-facing API.
