# Benchmark Baseline

`baseline.json` was generated on 2026-08-14 with:

```console
wmfs-benchmark --plugin-directory plugins --format json \
  --output benchmarks/baseline.json
```

The runtime and worker both used Python 3.14.6, Torch 2.12.0, and glibc 2.42.
Torch was limited to one CPU thread. Each operation was warmed up twice, then
measured ten times. Local and isolated outputs were retained for the duration of
each case so allocator reuse did not bias one side. The table reports medians;
the JSON report also contains p95, standard deviation, and transport
diagnostics.

| Operation | Size        | Local (ms) | Isolated (ms) | Overhead (ms) | Overhead |
| --------- | ----------- | ---------: | ------------: | ------------: | -------: |
| matmul    | 64 x 64     |      0.021 |         1.595 |         1.574 |  7465.9% |
| matmul    | 256 x 256   |      0.365 |         2.093 |         1.729 |   474.1% |
| matmul    | 2048 x 2048 |    128.534 |       133.161 |         4.627 |     3.6% |
| SVD       | 32 x 32     |      0.121 |         2.636 |         2.515 |  2077.5% |
| SVD       | 128 x 128   |      0.980 |         3.922 |         2.942 |   300.3% |
| SVD       | 768 x 768   |     67.902 |        74.073 |         6.171 |     9.1% |

The baseline shows the intended behavior: process-isolation overhead dominates
small operations, while it falls to 3.6% for the large matrix multiplication
and 9.1% for the large SVD. The cheap `add_scalar` measurements in
`baseline.json` expose the fixed control-plane cost directly.

These values characterize one WSL2 host and are not performance thresholds.
Regenerate the report on the target system when evaluating deployment choices.
