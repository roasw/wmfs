# Benchmark Baselines

The reports were generated on 2026-08-14 with:

```console
wmfs-benchmark --plugin-directory plugins --memory-mode pooled \
  --format json --output benchmarks/baseline.json
wmfs-benchmark --plugin-directory plugins --memory-mode arena \
  --arena-bytes 268435456 --format json --output benchmarks/arena.json
```

The runtime and worker both used Python 3.14.6, Torch 2.12.0, and glibc 2.42.
Torch was limited to one CPU thread. Each operation was warmed up twice, then
measured ten times. Result destruction and safe-pool retirement/reset are
included in isolated end-to-end samples. The table reports medians; the JSON
reports also contain p95, standard deviation, allocation statistics, and
transport diagnostics.

| Mode   | Operation  | Size             | Local (ms) | Isolated (ms) | Overhead (ms) | Overhead |
| ------ | ---------- | ---------------- | ---------: | ------------: | ------------: | -------: |
| pooled | matmul     | 64 x 64          |      0.022 |         1.955 |         1.933 |  8666.6% |
| arena  | matmul     | 64 x 64          |      0.018 |         1.741 |         1.723 |  9658.7% |
| pooled | matmul     | 256 x 256        |      0.254 |         2.497 |         2.243 |   881.4% |
| arena  | matmul     | 256 x 256        |      0.232 |         1.863 |         1.632 |   704.6% |
| pooled | matmul     | 2048 x 2048      |    120.415 |       130.196 |         9.781 |     8.1% |
| arena  | matmul     | 2048 x 2048      |    124.707 |       124.603 |        -0.104 |    -0.1% |
| pooled | SVD        | 32 x 32          |      0.131 |         3.408 |         3.278 |  2511.0% |
| arena  | SVD        | 32 x 32          |      0.119 |         2.657 |         2.538 |  2129.6% |
| pooled | SVD        | 128 x 128        |      0.974 |         4.890 |         3.916 |   401.9% |
| arena  | SVD        | 128 x 128        |      1.000 |         3.360 |         2.360 |   236.0% |
| pooled | SVD        | 768 x 768        |     67.570 |        78.367 |        10.797 |    16.0% |
| arena  | SVD        | 768 x 768        |     67.159 |        75.002 |         7.842 |    11.7% |
| pooled | add_scalar | 4096 elements    |      0.026 |         1.695 |         1.670 |  6490.1% |
| arena  | add_scalar | 4096 elements    |      0.023 |         1.339 |         1.316 |  5789.9% |
| pooled | add_scalar | 1048576 elements |      0.187 |         3.283 |         3.096 |  1655.3% |
| arena  | add_scalar | 1048576 elements |      0.164 |         1.722 |         1.558 |   950.5% |

Safe pooling reached 88.9% to 93.8% hit rates and bounded each case to three or
five memfds. It preserves one-FD-per-buffer capabilities, so each reused
generation still incurs acknowledged worker retirement and a new FD mapping.
The arena reached 97.1% to 98.8% suballocation hit rates with one memfd and
reduced most control-heavy medians. The slightly negative large-matmul result is
measurement noise, not evidence that isolation makes the kernel faster.

These values characterize one WSL2 host and are not performance thresholds.
Regenerate both reports on the target system when evaluating the security and
performance tradeoff.
