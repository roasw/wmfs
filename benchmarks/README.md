# Benchmark Reports

Retained reports use explicit revision names:

- `benchmark-0.1.0.json`: packaged Release benchmark from tag `0.1.0`.
- `benchmark-a137cb1.json`: schema-11 report after native ring optimization and
  before removal of the public local backend.
- `benchmark-3b2c5c4.json`: schema-12 report before native host transport became
  mandatory.
- `benchmark-master.json`: schema-13 mandatory-native-client report.

The reports were generated on 2026-08-24 on the same WSL2 host with Python
3.14.6, Torch 2.12.0, glibc 2.42, float32 tensors, and one Torch thread. Each
operation was warmed up twice and measured ten times.

Generate a comparable report with:

```console
just benchmark-json benchmarks/benchmark-master.json
```

Use an explicit mode suffix when retaining an arena result:

```console
just benchmark-json benchmarks/benchmark-master-arena.json arena
```

The recipes use the packaged Release runtime and worker. `just benchmark` and
`just benchmark arena` render the same measurements interactively.

## Comparison Contract

Schema 9 through schema 13 primary backend samples stop at backend return.
Result destruction, buffer retirement, collection, and allocator reset are
measured separately. Cleanup-inclusive high-frequency throughput includes that
work.

`benchmark-0.1.0.json` measures the fixed Cap'n Proto RPC operation path after
dynamic output allocation support was added. `benchmark-a137cb1.json` preserves
the former local/bundled/isolated schema. `benchmark-3b2c5c4.json` preserves the
selectable-client schema. `benchmark-master.json` compares bundled Python,
bundled native, and mandatory-native-client isolated execution and includes
initialization, logging modes, ring pressure, direct/profiled calls, and known
versus dynamic output paths.

The reports are not interchangeable schemas. Compare fields using their
embedded `measurement_boundaries`, `comparison_contract`, and
`diagnostic_provenance` metadata rather than renaming old fields.

## Transport Result

| Revision | Transport probe | Median (ms) | p95 (ms) |
| -------- | --------------- | ----------: | -------: |
| 0.1.0    | RPC round trip  |       0.092 |    0.196 |
| master   | Ring round trip |       0.065 |    0.101 |

The native ring dispatcher is 30% faster at the median than the freshly
measured `0.1.0` RPC baseline. It publishes fixed-width records directly from
C++, consumes completions on a native dispatcher, and does not serialize ring
records through Python.

The ring result does not imply that the complete cheap-operation path is faster.
Sequential cleanup-inclusive `add_scalar` throughput is 2,367 calls/s in
`0.1.0` and 1,130 calls/s on master; with reusable `out=`, it is 2,848 calls/s
and 1,326 calls/s respectively. Mapping, allocation, reclamation, and Python
orchestration remain outside the ring probe and are the next relevant
optimization boundaries.

These measurements characterize one host and are not performance thresholds.
Regenerate explicitly named reports on the target system when evaluating the
isolation tradeoff.
