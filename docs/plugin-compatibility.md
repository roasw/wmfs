# Plugin Artifact Compatibility

Generated plugin artifacts are deployment inputs, not caches. The runtime must
continue to consume artifacts emitted by a supported older generator without
regenerating them with the installed `wmfs-tool`. Accordingly, its `--check`
option only checks artifacts against the same tool version that generated the
deployment; it is not a runtime compatibility requirement.

Format v1 and plugin ABI v1 use exact integer versions. The v1 manifest is a
closed schema: its generator identifier and field sets must match exactly, and
unknown minor-version or feature fields are rejected with a diagnostic. The
runtime never guesses how to interpret an unknown field. A future format may
define explicit minor-version and feature negotiation, but that behavior must
not be retrofitted silently onto v1.

The immutable v1 snapshot under
`tests/integration/fixtures/generated-v1` is the compatibility baseline. It has
no source interface and must not be regenerated or compared to current live
generated files. Compatibility tests consume the manifest, Python metadata and
stub, C ABI header, and C++11 wrapper/stub directly from that snapshot.

Additive source-interface changes require regeneration and a format/ABI policy
that explicitly permits the new fields. Incompatible record or plugin ABI
changes require a new major version and a startup diagnostic. Cap'n Proto is not
part of the stable per-operation ABI: it is currently a startup/control
implementation detail. The stable operation boundary is the generated plugin
ABI plus the fixed-width ring format.

## GCC 4.8 compatibility boundary

Generated plugin-facing C ABI headers, C++11 wrappers, and generated dispatch
stubs are compiled in CI with the CentOS 7 system GCC 4.8.5. The check covers
the current reference artifacts, the mode-neutral fixture, and the immutable
generated-v1 fixture, including dtype dispatch, configuration declarations,
lifecycle hooks, and logging declarations. It deliberately does not build the
reference worker, numerical implementation, runtime, or other private native
code; those components use the project's normal modern compiler baseline.

`tests/compatibility/gcc48_compile.sh` uses `/usr/bin/gcc` for C99/C11 probes
and `/usr/bin/g++ -std=c++11 -Wall -Wextra -Werror` for generated C++ artifacts.
It fails unless both system compilers report exactly version 4.8.5. The GitHub
Actions job runs it in the maintained `manylinux2014_x86_64` image without
installing packages from the retired CentOS 7 network repositories.
