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
