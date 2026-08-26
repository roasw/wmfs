# Benchmark Reports

Retained reports use explicit revision names:

- `benchmark-0.1.0.json`: packaged Release benchmark from tag `0.1.0`.
- `benchmark-a137cb1.json`: schema-11 report after native ring optimization and
  before removal of the public local backend.
- `benchmark-3b2c5c4.json`: schema-12 report before native host transport became
  mandatory.
- `benchmark-8209176.json`: schema-13 report before whole-call scheduler and
  mapping-lifetime optimization.
- `benchmark-master.json`: optimized schema-13 mandatory-native-client report.

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
| master   | Ring round trip |       0.061 |    0.117 |

The native ring dispatcher is 34% faster at the median than the freshly
measured `0.1.0` RPC baseline. It publishes fixed-width records directly from
C++, consumes completions on a native dispatcher, and does not serialize ring
records through Python.

Whole-call profiling identified repeated asyncio/executor handoffs and a
redundant acknowledged output-mapping retirement as the dominant costs. Running
native operations synchronously on caller threads, batching known mappings,
locally completing validated mappings, and using a compact unprofiled completion
reduced allocated `add_scalar` median latency from 0.734 ms to 0.281 ms and
reusable-`out=` latency from 0.702 ms to 0.252 ms. Cleanup-inclusive throughput
rose from 1,130 to 2,877 calls/s and from 1,326 to 3,499 calls/s respectively.
Both optimized throughput values exceed the freshly measured `0.1.0` RPC path.

These measurements characterize one host and are not performance thresholds.
Regenerate explicitly named reports on the target system when evaluating the
isolation tradeoff.
