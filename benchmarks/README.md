# Benchmark Baselines

The reports were generated on 2026-08-15 with:

```console
wmfs-benchmark --plugin-directory plugins --memory-mode pooled \
  --control-mode native --high-frequency-iterations 1000 \
  --format json --output benchmarks/baseline.json
wmfs-benchmark --plugin-directory plugins --memory-mode arena \
  --arena-bytes 268435456 --control-mode native \
  --high-frequency-iterations 1000 --format json --output benchmarks/arena.json
```

The runtime used Python 3.14.6 and Torch 2.12.0. The C++ worker contained no
Python runtime and linked directly to LibTorch 2.12.0. Both used glibc 2.42.
Torch was limited to one CPU thread. Each operation was warmed up twice, then
measured ten times. Result destruction and safe-pool retirement/reset are
included in isolated end-to-end samples. Known outputs are preallocated from
schema metadata and each operation uses one Cap'n Proto RPC between the C++
client and C++ worker. The table reports medians; the JSON reports also contain
p95, standard deviation, allocation statistics, and transport diagnostics.

| Mode   | Operation  | Size             | Local (ms) | Isolated (ms) | Overhead (ms) | Overhead |
| ------ | ---------- | ---------------- | ---------: | ------------: | ------------: | -------: |
| pooled | matmul     | 64 x 64          |      0.018 |         0.349 |         0.331 |  1848.3% |
| arena  | matmul     | 64 x 64          |      0.021 |         0.339 |         0.318 |  1506.0% |
| pooled | matmul     | 256 x 256        |      0.238 |         0.689 |         0.451 |   189.0% |
| arena  | matmul     | 256 x 256        |      0.232 |         0.523 |         0.290 |   125.0% |
| pooled | matmul     | 2048 x 2048      |    125.291 |       131.127 |         5.837 |     4.7% |
| arena  | matmul     | 2048 x 2048      |    123.238 |       125.040 |         1.802 |     1.5% |
| pooled | SVD        | 32 x 32          |      0.115 |         0.628 |         0.512 |   444.8% |
| arena  | SVD        | 32 x 32          |      0.109 |         0.476 |         0.367 |   335.5% |
| pooled | SVD        | 128 x 128        |      1.028 |         1.822 |         0.793 |    77.1% |
| arena  | SVD        | 128 x 128        |      1.006 |         1.423 |         0.417 |    41.5% |
| pooled | SVD        | 768 x 768        |     70.120 |        75.199 |         5.079 |     7.2% |
| arena  | SVD        | 768 x 768        |     66.162 |        71.376 |         5.214 |     7.9% |
| pooled | add_scalar | 4096 elements    |      0.020 |         0.398 |         0.377 |  1864.9% |
| arena  | add_scalar | 4096 elements    |      0.017 |         0.200 |         0.183 |  1083.5% |
| pooled | add_scalar | 1048576 elements |      0.178 |         2.002 |         1.824 |  1025.4% |
| arena  | add_scalar | 1048576 elements |      0.170 |         0.622 |         0.452 |   265.7% |

Safe pooling reached 91.2% hit rates and bounded each case to three or five
memfds. It preserves one-FD-per-buffer capabilities, so each reused generation
still incurs acknowledged worker retirement and a new FD mapping. The arena
reached 97.1% suballocation hit rates with one memfd.

RPC-only median latency was 0.087 ms pooled and 0.090 ms in the arena. The
1,000-call high-frequency `add_scalar` run measured 0.324 ms median and 2,545
calls/s pooled, versus 0.202 ms and 4,440 calls/s in the arena. Reusing a managed
`out=` tensor reduced these to 0.246 ms and 3,294 calls/s pooled, and 0.158 ms
and 5,608 calls/s in the arena. These are sequential synchronous calls, not
batched operations.

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
high-frequency cheap-operation median by roughly 22-24% in this report.

These values characterize one WSL2 host and are not performance thresholds.
Regenerate both reports on the target system when evaluating the security and
performance tradeoff.
