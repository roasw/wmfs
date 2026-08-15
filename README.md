# wmfs

`wmfs` is a prototype scientific-computing runtime for transparently running
selected Python function calls in isolated worker processes. Its low-latency
control path uses a C++20 runtime bound with nanobind, Cap'n Proto C++ RPC, and
shared CPU tensors while workers may use an independently deployed Python and
Torch environment.

## Development

Enter the Nix development shell and run the tests:

```console
nix develop
cmake -S . -B build/Debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug
nix build
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
schema. Operation signatures, numeric IDs, and known output shape/dtype
expressions are declared once in the schema and discovered over RPC when the
plugin is registered:

```python
from pathlib import Path

from wmfs import runtime

runtime.discover_plugins(Path("plugins"))
print(runtime.operation_names)
```

Discovery exchanges metadata only. Discovered operations continue to execute
through the explicitly selected backend until isolated operation dispatch is
enabled.

Known output plans may reference input axes, minimum dimensions, Boolean scalar
selection, input dtypes, and tensor/scalar dtype promotion. The runtime
validates these plans during discovery, preallocates every known output, and
passes its writable descriptor in the operation request. This removes the
reverse output-allocator RPC from high-frequency calls. Dynamic operations may
still use the allocator capability.

## Shared CPU Tensors

The runtime can move a contiguous CPU tensor into runtime-owned memfd storage.
Workers receive a read-only descriptor through `SCM_RIGHTS`, map it once, and
construct a Torch view from the mapped memory. Cap'n Proto carries only tensor
metadata; numerical payload bytes never enter the RPC message.

Ordinary Torch allocations require one ingress copy into managed storage.
Managed results can be reused across worker calls without copying or repeatedly
passing the same FD.

The default memory mode pools whole memfds by exact size. A buffer returns to
the pool only after its last Torch storage alias is gone, all worker mappings
have acknowledged retirement, and its generation has advanced. Read-only input
mappings may remain cached while the allocation is live; writable output
mappings are scoped to one invocation. Pool limits bound idle FDs and bytes.

For trusted plugins, an optional arena mode suballocates one writable memfd and
maps it once per worker:

```python
runtime.configure_memory("arena", arena_bytes=256 * 1024 * 1024)
runtime.discover_plugins(Path("plugins"))
```

Configuration must happen before plugin discovery. Arena mode minimizes FD
passing and mapping overhead, but the worker can access every live allocation in
the arena. Use the default `pooled` mode when per-buffer capability boundaries
matter.

## Isolated Execution

After discovery, selecting the isolated backend starts a persistent worker on
first use. The worker writes operation results directly into storage allocated
by the runtime:

```python
from pathlib import Path

from wmfs import matmul, runtime

runtime.discover_plugins(Path("plugins"))
runtime.use_backend("isolated")
result = matmul(a, b)
runtime.close()
```

The default `auto` control mode uses the native extension when it is installed.
Selection can be made explicit before discovery:

```python
runtime.configure_control("native")  # or "python"
runtime.discover_plugins(Path("plugins"))
```

The native session owns synchronous Cap'n Proto/KJ dispatch and SCM_RIGHTS
mapping control on a dedicated C++ thread. The Python layer remains the public
Torch API and evaluates output metadata. Neither the native extension nor the
main process loads plugin code or links against the worker's Torch runtime.

`matmul`, `svd`, and `add_scalar` expose the same public API in local and
isolated modes. The current prototype supports contiguous CPU tensors and
serializes calls within each worker. Repeated calls reuse the persistent RPC
connection and cached arena or read-only pooled mappings.

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
wmfs-benchmark --plugin-directory plugins --control-mode native
```

The default run covers small, medium, and large inputs for `matmul`, `svd`, and
the deliberately cheap `add_scalar` operation. It reports median and standard
deviation for local kernel execution and isolated end-to-end execution, plus
absolute and percentage overhead. JSON output also records nearest-rank p95 and
a 1,000-call sequential high-frequency `add_scalar` latency/throughput run.

Separate diagnostics report worker startup, RPC-only round trips, shared-memory
allocation, uncached input preparation, first-use FD passing and worker mapping,
cached mapping checks, and runtime-owned output allocation. Input preparation
includes memfd allocation, the runtime mapping and Torch view, and the ingress
copy. Ensure-mapped timings include native dispatch, FD transfer, worker
mapping, and acknowledgement. Numerical-library warmup and worker startup are
excluded from steady-state operation timings.

Profiled invocations additionally separate output-plan evaluation, C++ queue
wait, RPC, worker input/output view construction, worker dispatch, and kernel
execution. Profiling showed repeated worker view construction was the largest
avoidable cheap-operation cost, so each worker mapping now keeps a bounded cache
of validated Torch views. Ordinary calls leave profiling disabled.

Write a machine-readable report with:

```console
wmfs-benchmark --plugin-directory plugins --format json --output benchmark.json
```

Compare the trusted single-FD arena with:

```console
wmfs-benchmark --plugin-directory plugins --memory-mode arena \
  --arena-bytes 268435456 --format json --output arena.json
```

Sizes, iteration counts, dtype, Torch thread count, control mode, and
high-frequency iteration count are configurable; run `wmfs-benchmark --help`
for all options. The output allocation service metric measures metadata-driven
runtime allocation and output mapping before the single operation RPC. Lazy
page faults remain part of isolated end-to-end time.
The checked-in [`benchmarks/baseline.json`](benchmarks/baseline.json) records a
complete safe-pool run, [`benchmarks/arena.json`](benchmarks/arena.json) records
the trusted-arena comparison, and
[`benchmarks/README.md`](benchmarks/README.md) summarizes their primary results.
