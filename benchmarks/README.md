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
| pooled | matmul     | 64 x 64          |      0.017 |         0.307 |         0.290 |  1747.9% |
| arena  | matmul     | 64 x 64          |      0.013 |         0.218 |         0.205 |  1518.1% |
| pooled | matmul     | 256 x 256        |      0.235 |         0.692 |         0.457 |   194.6% |
| arena  | matmul     | 256 x 256        |      0.246 |         0.481 |         0.235 |    95.7% |
| pooled | matmul     | 2048 x 2048      |    124.883 |       129.748 |         4.865 |     3.9% |
| arena  | matmul     | 2048 x 2048      |    124.262 |       126.562 |         2.299 |     1.9% |
| pooled | SVD        | 32 x 32          |      0.114 |         0.625 |         0.511 |   447.9% |
| arena  | SVD        | 32 x 32          |      0.112 |         0.414 |         0.303 |   270.6% |
| pooled | SVD        | 128 x 128        |      0.967 |         1.757 |         0.790 |    81.7% |
| arena  | SVD        | 128 x 128        |      0.974 |         1.352 |         0.378 |    38.8% |
| pooled | SVD        | 768 x 768        |     67.978 |        75.557 |         7.579 |    11.1% |
| arena  | SVD        | 768 x 768        |     69.761 |        72.992 |         3.231 |     4.6% |
| pooled | add_scalar | 4096 elements    |      0.019 |         0.308 |         0.290 |  1549.8% |
| arena  | add_scalar | 4096 elements    |      0.017 |         0.198 |         0.181 |  1085.5% |
| pooled | add_scalar | 1048576 elements |      0.204 |         2.207 |         2.003 |   984.4% |
| arena  | add_scalar | 1048576 elements |      0.180 |         0.644 |         0.465 |   258.3% |

Safe pooling reached 91.2% hit rates and bounded each case to three or five
memfds. It preserves one-FD-per-buffer capabilities, so each reused generation
still incurs acknowledged worker retirement and a new FD mapping. The arena
reached 97.1% suballocation hit rates with one memfd.

RPC-only median latency was 0.086 ms in both modes. The 1,000-call
high-frequency `add_scalar` run measured 0.303 ms median and 2,791 calls/s
pooled, versus 0.186 ms and 4,585 calls/s in the arena. These are sequential
synchronous calls, not batched operations.

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

These values characterize one WSL2 host and are not performance thresholds.
Regenerate both reports on the target system when evaluating the security and
performance tradeoff.
