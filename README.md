# wmfs

`wmfs` is a prototype scientific-computing runtime for transparently running
selected Python function calls in isolated worker processes. Its low-latency
path uses a C++20 runtime bound with nanobind, Cap'n Proto C++ RPC, shared CPU
tensors, and an independently deployed C++ worker linked to its own LibTorch
environment.

## Development Build

The `wmfs` Python distribution lives under `packages/wmfs`; root-level CMake,
C++ sources, tests, plugins, Nix definitions, and benchmarks remain shared
repository infrastructure.

The development shell inherits build inputs from the runtime and worker package
derivations with `inputsFrom`. This keeps the CMake and package builds on the
same Python, Cap'n Proto, nanobind, Torch, compiler, and linker dependencies.

The shell provides a `justfile` that configures one CMake tree for both the
native runtime extension and C++ reference worker, then installs the runnable
artifacts into the matching ignored output directory:

```console
nix develop
just --list
just debug
```

This keeps generated CMake state under `build/Debug`, `build/Release`, and so
on, while installed artifacts live under the corresponding `output/Debug` or
`output/Release` prefix. Build another configuration explicitly or select the
default before entering the shell:

```console
just release
just build RelWithDebInfo
WMFS_BUILD_TYPE=Release nix develop
```

The shell adds the selected output prefix and `packages/wmfs` source directory
to `PYTHONPATH`, and adds the output prefix to `PATH`. Re-run the corresponding
`just build` recipe after source changes. Verify that the development artifacts
are selected, then run tests directly from the source tree:

```console
python -c 'import wmfs._native; print(wmfs._native.__file__)'
command -v wmfs-reference-worker
wmfs-reference-worker --help
just test
just test-release
```

The worker is normally launched by the runtime with private RPC and FD-passing
descriptors; `--help` only verifies the executable outside an invocation.

## Release Build

Release artifacts are produced by the Nix packages. They configure CMake in
Release mode and package the runtime and workers independently:

```console
just package
just check
just check-pinned
```

The `bundled` package compiles the reference plugin into an optional
in-process extension against the application's LibTorch:

```console
nix build .#bundled
```

For development builds, `just build` configures
`WMFS_BUNDLED_PLUGINS=reference`. Direct CMake builds can leave the list empty
for an isolation-only runtime or provide a semicolon-separated build-time
plugin list. Runtime discovery cannot add code to an already compiled bundle.

Use `nix shell`, rather than a development build, when measuring packaged
Release performance:

```console
just benchmark
just benchmark arena
```

Reusable package derivations live under `nix/`; the root `flake.nix` only wires
packages, checks, and development shells together.

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

The `local` backend is the direct Torch baseline and remains the default. A
runtime built with bundled plugins also exposes `bundled` without changing the
public calls:

```python
runtime.use_backend("bundled")
c = matmul(a, b)
```

The bundled extension is loaded lazily on its first invocation. It calls the
same transport-neutral C++ kernels as the isolated worker but does not create a
worker, allocate shared memory, transfer file descriptors, or use Cap'n Proto.
Bundled plugins must use the application's compiler ABI, glibc, LibTorch, and
dependency versions, and a plugin failure can terminate the application.

The direct `local` backend is retained both as the benchmark reference and for
Torch's unrestricted native operation semantics. Bundled dispatch has a small
fixed custom-operator cost, so replacing `local` would not improve the existing
three built-in operations.

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

Runtime-mediated access is reserved atomically for each operation. Read-only
inputs may be used by multiple workers concurrently. Explicitly mutable inputs
and reusable outputs receive an exclusive write lease until the operation has
completed, or until a failed worker has been stopped. Aliases share one lease,
and an operation acquires its complete access set at once, so it never holds one
buffer while waiting for another. Queued writers are not bypassed by later
readers of the same allocation.

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
main process loads the worker or links against the worker's Torch runtime.
Selecting isolated execution does not load the optional bundled extension;
applications that previously invoked bundled code have already opted out of
process-level plugin isolation for that code.

The packaged reference worker implements the server control plane in C++20 and
constructs ATen tensor views directly over mapped memfds. It executes the
reference kernels with LibTorch, removing pycapnp, asyncio, Python descriptor
conversion, and Python-to-Torch dispatch from the worker hot path. The previous
Python implementation remains available as the `reference-python-worker` Nix
package for comparison and fallback testing.

`matmul`, `svd`, and `add_scalar` expose the same public API in local and
isolated modes. The current prototype supports contiguous CPU tensors and
serializes calls within each worker. Repeated calls reuse the persistent RPC
connection and cached arena or read-only pooled mappings.

Like PyTorch, these operations accept an optional `out=` argument. Local mode
accepts an ordinary compatible Torch tensor. Isolated mode requires a live
managed result from the same runtime so the worker can write it without a copy:

```python
result = add_scalar(a, 0.0)
for value in values:
    add_scalar(a, value, out=result)
```

Reusable isolated outputs must have the schema-derived shape and dtype, cannot
require gradients, and cannot alias an input or another output. Existing calls
without `out=` retain the allocate-and-return behavior.

## Incompatible Worker Environment

`environments/nixos-25.05` is an independent nested flake that rebuilds the
reference worker with NixOS 25.05, glibc 2.40, Cap'n Proto 1.1, and LibTorch
2.7. The worker contains no Python runtime. The root runtime remains built from
its separately pinned unstable Nixpkgs input.

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

Run the local-versus-isolated benchmark with the packaged Release runtime and
worker:

```console
just benchmark
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
execution. Profiling first showed repeated worker view construction was the
largest avoidable cheap-operation cost, so each worker mapping keeps a bounded
cache of validated tensor views. Moving the worker control plane and view
construction to C++ reduced the remaining Python worker scheduling overhead.
Ordinary calls leave profiling disabled.

Protocol v7 uses separate ordinary and profiled RPC methods, so ordinary calls
carry no metrics result. The native session caches value-only tensor descriptors
and uses an allocation-free synchronous handoff to its thread-affine KJ event
loop. At this point process scheduling and the required RPC completion dominate
cheap calls; larger improvements require output reuse, batching, or changing the
eager execution model.

Write a machine-readable report with:

```console
just benchmark-json benchmark.json
```

Compare the trusted single-FD arena with:

```console
just benchmark-json arena.json arena
```

Sizes, iteration counts, dtype, Torch thread count, control mode, and
high-frequency iteration count are configurable; run `just benchmark-help` for
all underlying options. The output allocation service metric measures
metadata-driven runtime allocation and output mapping before the single
operation RPC. Lazy page faults remain part of isolated end-to-end time.
The checked-in [`benchmarks/baseline.json`](benchmarks/baseline.json) records a
complete safe-pool run, [`benchmarks/arena.json`](benchmarks/arena.json) records
the trusted-arena comparison, and
[`benchmarks/README.md`](benchmarks/README.md) summarizes their primary results.
