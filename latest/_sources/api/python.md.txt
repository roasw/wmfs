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
do not depend on `wmfs-tool` at runtime. Every Python plugin distribution
depends on `wmfs-plugin`; C++ workers do not.

## Runtime

```{eval-rst}
.. autoclass:: wmfs.runtime.Runtime
   :members:

.. autoclass:: wmfs.transport.deadlines.TransportDeadlines
   :members:

.. autoclass:: wmfs.configuration.ConfigurationMetadata
   :members:

.. autoclass:: wmfs.logging.LoggingOptions
   :members:
```

`Runtime.load_plugins` is the worker-free manifest path used before
`list_configurable`, `validate_config`, and `configure_plugin`. Configuration is
canonicalized without inserting defaults and becomes immutable at plugin
initialization. Bundled plugins initialize when selected; isolated
plugins initialize during eager discovery. `Runtime.close` invokes enabled
shutdown hooks and resets lifecycle state.

`LoggingOptions()` selects the zero-resource disabled path. Centralized mode
uses an independent bounded log channel and worker-file mode uses an independent
bounded local sink; neither shares operation or FD-control traffic.

## Plugin SDK

`wmfs-plugin` is documented and tested as an independent Python plugin SDK.
The core runtime deliberately does not import it; Python plugin distributions
bring the facade and compatible worker protocol definitions into bundled or
isolated environments. See `packages/wmfs-plugin/README.md` and generated
Python interface stubs.
