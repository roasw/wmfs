# wmfs-tool

`wmfs-tool` compiles a versioned TOML plugin interface into deterministic
runtime metadata and C/C++11 and Python plugin-facing artifacts. It is an
independent build-time package and does not depend on the WMFS runtime, plugin
SDK, PyTorch, or Cap'n Proto.

```console
wmfs-tool generate --interface interface.toml --output generated
wmfs-tool generate --interface interface.toml --output generated --check
```

The interface file is the source of truth. Fingerprints are derived by the
compiler and must not be written into the source TOML. `--check` exits with a
useful error when generated files are missing or stale.

For the reference plugin the same invocation also owns the generated Python
worker adapter and C++ worker dispatch include. Generated Python worker code may
import `wmfs_plugin`; the generator itself has no SDK dependency.
