# wmfs

`wmfs` is a prototype scientific-computing runtime for transparently running
selected Python function calls in isolated worker processes. The first
milestone provides the local PyTorch backend that will serve as the behavioral
and performance baseline for isolated execution.

## Development

Enter the Nix development shell and run the tests:

```console
nix develop
pytest
```

## Usage

```python
import torch

from wmfs import add_scalar, matmul, runtime, svd

a = torch.randn(4, 4)
b = torch.randn(4, 4)

runtime.use_backend("local")
c = matmul(a, b)
u, s, vh = svd(c)
d = add_scalar(c, 1.0)
```

The public calls will remain unchanged when the isolated backend is added.

## Plugin Discovery

Plugin deployment metadata identifies a worker module and its Cap'n Proto
schema. Operation signatures are declared once in the schema and discovered
over RPC when the plugin is registered:

```python
from pathlib import Path

from wmfs import runtime

runtime.discover_plugins(Path("plugins"))
print(runtime.operation_names)
```

Discovery exchanges metadata only. Discovered operations continue to execute
through the explicitly selected backend until isolated operation dispatch is
enabled.

## Shared CPU Tensors

The runtime can move a contiguous CPU tensor into runtime-owned memfd storage.
Workers receive a read-only descriptor through `SCM_RIGHTS`, map it once, and
construct a Torch view from the mapped memory. Cap'n Proto carries only tensor
metadata; numerical payload bytes never enter the RPC message.

Ordinary Torch allocations require one ingress copy into managed storage.
Managed results can be reused across worker calls without copying or repeatedly
passing the same FD.

## Isolated Execution

After discovery, selecting the isolated backend starts a persistent worker on
first use. Inputs and runtime-owned outputs are mapped once per worker, and the
worker writes operation results directly into storage allocated by the runtime:

```python
from pathlib import Path

from wmfs import matmul, runtime

runtime.discover_plugins(Path("plugins"))
runtime.use_backend("isolated")
result = matmul(a, b)
runtime.close()
```

`matmul`, `svd`, and `add_scalar` expose the same public API in local and
isolated modes. The current prototype supports contiguous CPU tensors and
serializes calls within each worker.

## Incompatible Worker Environment

`environments/nixos-25.05` is an independent nested flake that rebuilds the
reference worker with NixOS 25.05, glibc 2.40, Python 3.12, and its own Torch
closure. The root runtime remains built from its separately pinned unstable
Nixpkgs input.

Run the cross-environment integration check from the repository root:

```console
nix flake check ./environments/nixos-25.05
```

The check confirms the runtime and worker report different glibc versions and
then executes an isolated tensor operation through Cap'n Proto and shared
memory. It also verifies that plugin modules and native libraries from the old
worker closure are not loaded into the main process. Enter the old worker
development shell with:

```console
nix develop ./environments/nixos-25.05
```

## Benchmarking

Run the local-versus-isolated benchmark from the development shell:

```console
wmfs-benchmark --plugin-directory plugins
```

The default run covers small, medium, and large inputs for `matmul`, `svd`, and
the deliberately cheap `add_scalar` operation. It reports median and standard
deviation for local kernel execution and isolated end-to-end execution, plus
absolute and percentage overhead. JSON output also records nearest-rank p95.

Separate diagnostics report worker startup, RPC-only round trips, shared-memory
allocation, uncached input preparation, first-use FD passing and worker mapping,
cached mapping checks, and runtime-owned output allocation. Input preparation
includes memfd allocation, the runtime mapping and Torch view, and the ingress
copy. Ensure-mapped timings include the event-loop handoff, FD transfer, worker
mapping, and acknowledgement. Numerical-library warmup and worker startup are
excluded from steady-state operation timings.

Write a machine-readable report with:

```console
wmfs-benchmark --plugin-directory plugins --format json --output benchmark.json
```

Sizes, iteration counts, dtype, and Torch thread count are configurable; run
`wmfs-benchmark --help` for all options. The output allocator service metric
measures runtime allocation and output FD mapping inside the callback handler;
lazy page faults and callback transit remain part of isolated end-to-end time.
The checked-in [`benchmarks/baseline.json`](benchmarks/baseline.json) records a
complete reference run and [`benchmarks/README.md`](benchmarks/README.md)
summarizes its primary results.
