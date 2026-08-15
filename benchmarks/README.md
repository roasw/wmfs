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
| pooled | matmul     | 64 x 64          |      0.020 |         0.670 |         0.650 |  3255.3% |
| arena  | matmul     | 64 x 64          |      0.019 |         0.454 |         0.435 |  2268.3% |
| pooled | matmul     | 256 x 256        |      0.255 |         1.119 |         0.864 |   339.4% |
| arena  | matmul     | 256 x 256        |      0.255 |         0.692 |         0.437 |   171.7% |
| pooled | matmul     | 2048 x 2048      |    130.738 |       137.808 |         7.070 |     5.4% |
| arena  | matmul     | 2048 x 2048      |    121.160 |       121.562 |         0.402 |     0.3% |
| pooled | SVD        | 32 x 32          |      0.118 |         1.041 |         0.923 |   781.9% |
| arena  | SVD        | 32 x 32          |      0.118 |         0.655 |         0.537 |   457.1% |
| pooled | SVD        | 128 x 128        |      0.985 |         2.258 |         1.273 |   129.1% |
| arena  | SVD        | 128 x 128        |      0.979 |         1.571 |         0.591 |    60.4% |
| pooled | SVD        | 768 x 768        |     68.557 |        76.460 |         7.903 |    11.5% |
| arena  | SVD        | 768 x 768        |     66.664 |        71.723 |         5.059 |     7.6% |
| pooled | add_scalar | 4096 elements    |      0.025 |         0.674 |         0.649 |  2598.9% |
| arena  | add_scalar | 4096 elements    |      0.021 |         0.474 |         0.453 |  2170.3% |
| pooled | add_scalar | 1048576 elements |      0.179 |         2.167 |         1.988 |  1112.7% |
| arena  | add_scalar | 1048576 elements |      0.157 |         0.596 |         0.439 |   279.4% |

Safe pooling reached 91.2% hit rates and bounded each case to three or five
memfds. It preserves one-FD-per-buffer capabilities, so each reused generation
still incurs acknowledged worker retirement and a new FD mapping. The arena
reached 97.1% suballocation hit rates with one memfd.

RPC-only median latency was about 0.126 ms in both modes. The 1,000-call
high-frequency `add_scalar` run measured 0.575 ms median and 1,525 calls/s
pooled, versus 0.360 ms and 2,391 calls/s in the arena. These are sequential
synchronous calls, not batched operations.

These values characterize one WSL2 host and are not performance thresholds.
Regenerate both reports on the target system when evaluating the security and
performance tradeoff.
