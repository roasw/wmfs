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

The runtime and worker both used Python 3.14.6, Torch 2.12.0, and glibc 2.42.
Torch was limited to one CPU thread. Each operation was warmed up twice, then
measured ten times. Result destruction and safe-pool retirement/reset are
included in isolated end-to-end samples. Known outputs are preallocated from
schema metadata and each operation uses one Cap'n Proto RPC through the C++
control path. The table reports medians; the JSON reports also contain p95,
standard deviation, allocation statistics, and transport diagnostics.

| Mode   | Operation  | Size             | Local (ms) | Isolated (ms) | Overhead (ms) | Overhead |
| ------ | ---------- | ---------------- | ---------: | ------------: | ------------: | -------: |
| pooled | matmul     | 64 x 64          |      0.021 |         0.611 |         0.591 |  2865.8% |
| arena  | matmul     | 64 x 64          |      0.018 |         0.315 |         0.297 |  1651.2% |
| pooled | matmul     | 256 x 256        |      0.232 |         1.117 |         0.885 |   381.9% |
| arena  | matmul     | 256 x 256        |      0.257 |         0.696 |         0.438 |   170.5% |
| pooled | matmul     | 2048 x 2048      |    122.500 |       129.135 |         6.635 |     5.4% |
| arena  | matmul     | 2048 x 2048      |    122.771 |       121.358 |        -1.412 |    -1.2% |
| pooled | SVD        | 32 x 32          |      0.117 |         1.039 |         0.922 |   788.6% |
| arena  | SVD        | 32 x 32          |      0.112 |         0.576 |         0.464 |   414.9% |
| pooled | SVD        | 128 x 128        |      1.002 |         2.212 |         1.209 |   120.6% |
| arena  | SVD        | 128 x 128        |      1.046 |         1.661 |         0.615 |    58.8% |
| pooled | SVD        | 768 x 768        |     67.513 |        74.341 |         6.828 |    10.1% |
| arena  | SVD        | 768 x 768        |     67.344 |        73.624 |         6.280 |     9.3% |
| pooled | add_scalar | 4096 elements    |      0.023 |         0.644 |         0.621 |  2693.6% |
| arena  | add_scalar | 4096 elements    |      0.021 |         0.431 |         0.409 |  1915.0% |
| pooled | add_scalar | 1048576 elements |      0.171 |         1.912 |         1.742 |  1020.6% |
| arena  | add_scalar | 1048576 elements |      0.163 |         0.593 |         0.430 |   263.5% |

Safe pooling reached 91.2% hit rates and bounded each case to three or five
memfds. It preserves one-FD-per-buffer capabilities, so each reused generation
still incurs acknowledged worker retirement and a new FD mapping. The arena
reached 97.1% suballocation hit rates with one memfd.

RPC-only median latency was 0.154 ms pooled and 0.132 ms in the arena. The
1,000-call high-frequency `add_scalar` run measured 0.559 ms median and 1,604
calls/s pooled, versus 0.342 ms and 2,473 calls/s in the arena. These are
sequential synchronous calls, not batched operations.
The negative large-matmul overhead is measurement noise.

The profile identified repeated worker tensor-view construction as the largest
avoidable cheap-operation cost. Caching validated views per worker mapping
reduced combined input/output view construction from about 0.070 ms to about
0.020 ms for small arena `add_scalar`. Detailed JSON diagnostics now separate
output-plan evaluation, native queue wait, RPC, worker views, dispatch, and
kernel execution. Profiling is opt-in on each invocation, so ordinary calls do
not execute the timing code.

These values characterize one WSL2 host and are not performance thresholds.
Regenerate both reports on the target system when evaluating the security and
performance tradeoff.
