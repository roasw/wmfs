# Benchmark Baselines

The reports were generated on 2026-08-15 with:

```console
just benchmark-json benchmarks/baseline.json
just benchmark-json benchmarks/arena.json arena
```

The recipes use the packaged Release runtime and worker. Use `just benchmark`
or `just benchmark arena` for the same measurements rendered interactively.

The runtime used Python 3.14.6 and Torch 2.12.0. The native runtime extension
and C++ worker were built in Release mode by their Nix packages. The worker
contained no Python runtime and linked directly to LibTorch 2.12.0. Both used
glibc 2.42.
Torch was limited to one CPU thread. Each operation was warmed up twice, then
measured ten times. The checked report envelopes use schema 8 and explicitly
contain no current operation samples because bundled measurements have not been
generated on the reference host. Their `historical_report` links point to the
earlier schema 5 numeric reports retained as `baseline.schema5.json` and
`arena.schema5.json`, rather than relabeling or fabricating results. Those
legacy boundaries included result destruction and safe-pool
retirement/reset in isolated end-to-end samples. New schema 8 reports contain
local, bundled, and isolated samples keyed by backend. All primary samples stop
at backend return and post-return cleanup is measured separately.
Known outputs are preallocated from schema metadata and each operation uses one
Cap'n Proto RPC between the C++ client and C++ worker. The table reports
medians; the JSON reports also contain p95, standard deviation, allocation
statistics, and transport diagnostics. Pooled reclamation diagnostics use
internal cumulative metric deltas and report the reclaimed-buffer population,
recipient retirement, and memfd reset time for each sample.

| Mode   | Operation  | Size             | Local (ms) | Isolated (ms) | Overhead (ms) | Overhead |
| ------ | ---------- | ---------------- | ---------: | ------------: | ------------: | -------: |
| pooled | matmul     | 64 x 64          |      0.018 |         0.349 |         0.331 |  1793.1% |
| arena  | matmul     | 64 x 64          |      0.015 |         0.279 |         0.264 |  1722.0% |
| pooled | matmul     | 256 x 256        |      0.237 |         0.893 |         0.655 |   275.9% |
| arena  | matmul     | 256 x 256        |      0.230 |         0.521 |         0.291 |   126.7% |
| pooled | matmul     | 2048 x 2048      |    122.849 |       130.373 |         7.524 |     6.1% |
| arena  | matmul     | 2048 x 2048      |    121.406 |       123.634 |         2.228 |     1.8% |
| pooled | SVD        | 32 x 32          |      0.114 |         0.776 |         0.662 |   578.3% |
| arena  | SVD        | 32 x 32          |      0.111 |         0.445 |         0.334 |   300.2% |
| pooled | SVD        | 128 x 128        |      0.993 |         1.743 |         0.750 |    75.5% |
| arena  | SVD        | 128 x 128        |      0.995 |         1.453 |         0.458 |    46.0% |
| pooled | SVD        | 768 x 768        |     66.120 |        73.866 |         7.746 |    11.7% |
| arena  | SVD        | 768 x 768        |     63.988 |        70.318 |         6.330 |     9.9% |
| pooled | add_scalar | 4096 elements    |      0.019 |         0.412 |         0.392 |  2033.1% |
| arena  | add_scalar | 4096 elements    |      0.018 |         0.266 |         0.248 |  1386.7% |
| pooled | add_scalar | 1048576 elements |      0.177 |         2.001 |         1.824 |  1030.4% |
| arena  | add_scalar | 1048576 elements |      0.187 |         0.645 |         0.459 |   245.7% |

Safe pooling reached 91.2% hit rates and bounded each case to three or five
memfds. It preserves one-FD-per-buffer capabilities, so each reused generation
still incurs acknowledged worker retirement and a new FD mapping. The arena
reached 97.1% suballocation hit rates with one memfd.

RPC-only median latency was 0.082 ms pooled and 0.081 ms in the arena. The
schema 5 1,000-call high-frequency `add_scalar` run measured 0.337 ms median and
2,502 calls/s pooled, versus 0.226 ms and 4,107 calls/s in the arena. Reusing a
managed `out=` tensor reduced these to 0.262 ms and 3,186 calls/s pooled, and
0.183 ms and 4,675 calls/s in the arena. Current reports label backend-return
call latency separately from cleanup-inclusive whole-loop throughput. These are
sequential synchronous calls, not batched operations.

The profile identified repeated worker tensor-view construction as the largest
avoidable cheap-operation cost. Caching validated views per worker mapping
reduced combined Python input/output view construction from about 0.070 ms to
about 0.020 ms. Implementing the worker control plane and ATen views in C++
reduced that to about 0.003 ms and removed pycapnp and asyncio from steady-state
dispatch. Compared with the optimized Python worker report, arena small
`add_scalar` fell from 0.431 ms to 0.198 ms and high-frequency latency fell from
0.342 ms to 0.186 ms. Detailed JSON diagnostics separate output-plan
evaluation, native queue wait, RPC, worker views, dispatch, and kernel execution.
Profiling is opt-in on each invocation, so ordinary calls do not execute the
timing code.

Protocol v7 separates ordinary completion from profiled metrics. The native
client also caches value-only tensor descriptors and replaces its allocating
promise/function queue with a synchronous semaphore handoff. These changes
reduce allocation pressure but did not move ordinary end-to-end latency beyond
run-to-run noise: the mandatory RPC and thread wakeup now dominate. Reusable
outputs are the measurable remaining eager-path optimization, improving the
high-frequency cheap-operation median by roughly 19-22% in this report.

These historical values characterize one WSL2 host and are not performance
thresholds. Regenerate both schema 8 reports on the target system when
evaluating the security and performance tradeoff.
