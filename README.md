# wmfs

Documentation: <https://roasw.github.io/wmfs/>

`wmfs` is a prototype scientific-computing runtime for transparently running
selected Python function calls in isolated worker processes. Its low-latency
path uses a C++20 runtime bound with nanobind, Cap'n Proto C++ RPC, shared CPU
tensors, and an independently deployed C++ worker linked to its own LibTorch
environment.

## Development Build

The `wmfs` Python distribution lives under `packages/wmfs`. The independent
worker-side SDK and protocol schemas live under `packages/wmfs-plugin`.
Root-level CMake, C++ sources, tests, plugins, Nix definitions, and benchmarks
remain shared repository infrastructure.

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

The shell adds the selected output prefix plus the `packages/wmfs` and
`packages/wmfs-plugin` source directories to `PYTHONPATH`, and adds the output
prefix to `PATH`. Re-run the corresponding `just build` recipe after source
changes. Verify that the development artifacts are selected, then run tests
directly from the source tree:

```console
python -c 'import wmfs._native; print(wmfs._native.__file__)'
command -v wmfs-reference-worker
wmfs-reference-worker --help
just test
just test-release
```

Build versions come from the current Git revision. CMake and documentation use
the short revision directly, for example `50c505e17aac-dirty`. Python package
metadata uses the equivalent PEP 440 form, such as
`0.0.0+g50c505e17aac.dirty`. Nix obtains the revision from `self.rev` or
`self.dirtyRev`; the development shell exports the same values for local
builds.

`just test` and `just test-all` build once and run every test layer. Use
`just test-unit`, `just test-contract`, `just test-integration`, `just test-sdk`,
`just test-native`, or `just test-package` to run one layer independently.

The worker is normally launched by the runtime with private RPC and FD-passing
descriptors; `--help` only verifies the executable outside an invocation.

## Documentation

The unified Sphinx site renders Python docstrings with autodoc, C++ Doxygen XML
with Breathe, and architecture guides written in MyST Markdown. Build it with
the reproducible default development shell:

```console
nix develop -c just doc
```

From an existing development shell, run `just doc`. This configures and builds
the normal CMake `doc` target. The generated site starts at
`build/Debug/docs/html/index.html` by default, or under the selected
`WMFS_BUILD_TYPE`. The data-flow guide follows a simple Python call through
runtime dispatch, shared-memory allocation, batched FD transfer, worker view
construction, kernel execution, and reclamation.

## Buffer Transport Protocol

Protocol v9 sends an ordered list of buffer map and retirement entries in one
`SOCK_SEQPACKET` control message. Map entries correspond positionally to one
`SCM_RIGHTS` descriptor array and the worker acknowledges the complete batch
once. Descriptor counts must match map entries exactly; received descriptors
are close-on-exec and every descriptor is closed if validation or mapping
fails. A read-only mapping upgrade is represented by an ordered retirement
followed by its writable map, preserving per-buffer and per-generation
capabilities.

Any rejected or malformed batch invalidates the transport mapping cache on
both sides. Metrics count mapping and retirement batches separately from the
number of mapped and retired buffers.

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
from wmfs import add_scalar, matmul, randn, runtime, svd

runtime.use_backend("local")
a = randn(4, 4)
b = randn(4, 4)
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

Python plugins depend on the standalone `wmfs-plugin` distribution rather than
the main runtime; the SDK source does not import `wmfs`. For v0.1 its invocation,
shared-memory transport, and worker layers deliberately target Torch CPU
tensors. The wire schemas and metadata model are Torch-independent and can be
imported by control-plane tooling without loading Torch. Both layers remain in
one SDK distribution until a separate package has a concrete use case.

The SDK provides the protocol schemas, FD receiver, mapped Torch views, and
metadata-driven worker bootstrap:

```python
from wmfs_plugin import InvocationContext, worker_main


def my_operation_handler(context: InvocationContext) -> None:
    context.output("result").copy_(context.input("value"))


worker_main({"my_operation": my_operation_handler})
```

The operation-scoped context carries metadata, an invocation ID, ordinary input
tensors, writable output tensors, and scalar values. Its `input()`, `output()`,
and `scalar()` accessors accept metadata names or positional indices. Plugin
kernels do not receive runtime object-store, mapping-cache, RPC, memfd, or
allocator internals. The reference Python worker under `plugins/reference`
demonstrates the complete adapter.

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

Discovery is eager: it starts one worker per plugin, validates its metadata, and
retains that session for isolated execution. A discovery failure closes every
session started by that attempt, and the existing registry remains unchanged.
Discovered operations continue to execute through the explicitly selected
backend until isolated operation dispatch is enabled.

Transport deadlines are immutable for a discovered backend and can be changed
before discovery. Defaults are 30 seconds for startup, requests, shutdown, and
post-terminate kill grace, and 5 seconds for FD transfer:

```python
runtime.configure_deadlines(
    startup=30, request=30, fd_transfer=5, shutdown=30, kill_grace=30
)
runtime.discover_plugins(Path("plugins"))
```

Output plans may reference input axes, minimum dimensions, Boolean scalar
selection, input dtypes, and tensor/scalar dtype promotion. The runtime
validates these plans during discovery, preallocates every output, and passes
its writable descriptor in the operation request. Dynamic output plans are
reserved by the protocol but rejected until a generic allocator invocation is
implemented for both control paths.

Plugins may advertise an internal vector-Jacobian product (VJP) operation for a
public operation. Its metadata identifies the forward inputs and outputs that
must be saved, output cotangents it consumes, scalar values it needs, and input
gradients it returns. The plugin implements the VJP as an ordinary tensor
operation using the same known-output and shared-memory protocol. Internal VJP
operations participate in registration and validation but are omitted from
`runtime.operation_names`.

## Shared CPU Tensors

The runtime can move a contiguous CPU tensor into runtime-owned memfd storage.
Workers receive a read-only descriptor through `SCM_RIGHTS`, map it once, and
construct a Torch view from the mapped memory. Cap'n Proto carries only tensor
metadata; numerical payload bytes never enter the RPC message.

Use `wmfs.empty`, `wmfs.zeros`, `wmfs.ones`, or `wmfs.randn` after selecting the
isolated backend to allocate and initialize inputs directly in shared storage.
They return ordinary Torch tensors whose storage carries a WMFS allocation
lease, so Torch operations and autograd continue to work normally. Local and
bundled modes delegate these constructors to Torch and do not allocate shared
memory or load the bundled extension.

Ordinary Torch allocations still require one ingress copy into managed storage.
WMFS-created tensors and managed results can be reused across worker calls
without copying. Shared constructors support CPU tensors with `float32`,
`float64`, `int64`, or `uint8` storage and require a non-empty shape with
positive dimensions.

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

from wmfs import matmul, randn, runtime

runtime.discover_plugins(Path("plugins"))
runtime.use_backend("isolated")
a = randn(4, 4)
b = randn(4, 4)
result = matmul(a, b)
runtime.close()
```

`runtime.close()` is idempotent and the runtime remains reusable. Close first
stops accepting invocations, waits for every invocation and plugin discovery
already accepted by the runtime, and then attempts to close every backend even
if one fails. Calls arriving while close is in progress fail with
`RuntimeError`; concurrent close calls wait for that close. After cleanup, the
runtime has the same state as a new instance: the local backend is selected,
the plugin registry is empty, and memory and control modes are `pooled` and
`auto`. Cleanup raises the first resource failure after attempting the rest,
but the reset still completes.

Managed tensors already returned by isolated operations remain valid after
close. Their shared storage is released only after the last Torch storage alias
dies; close does not invalidate live tensor views.

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

Isolated operations that advertise a VJP participate in ordinary PyTorch
autograd graphs. PyTorch schedules the graph in the main process; WMFS invokes
the plugin's forward and VJP operations in its isolated worker. The reference
plugin provides VJPs for `matmul` and `add_scalar`, so they can be chained with
local Torch operations and with each other before calling `backward()`. `svd`
currently raises when an isolated differentiable call is requested because it
does not advertise a VJP. The initial contract supports first-order reverse
mode only and rejects higher-order gradients, mutable differentiable inputs,
and `out=` with differentiable inputs.

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

Run the local, bundled, and isolated benchmark with the packaged Release
runtime, bundled reference plugin, and worker. The recipe uses the packaged
benchmark app, so it can be invoked from outside the checkout and does not use
source files from the current directory:

```console
just benchmark
```

Equivalently, run `nix run path:/path/to/wmfs#benchmark`. The app fixes the
plugin directory to the packaged reference worker and clears `PYTHONPATH`,
`PYTHONHOME`, `LD_PRELOAD`, and `LD_LIBRARY_PATH` before starting the packaged
bundled runtime. The raw installed `wmfs-benchmark` executable intentionally
requires an explicit `--plugin-directory`; it never defaults to `./plugins`.

The default run covers small, medium, and large inputs for `matmul`, `svd`, and
the deliberately cheap `add_scalar` operation. It instantiates and warms all
three backends, verifies equivalent results, and rotates their measurement
order. It reports backend-keyed median and standard deviation plus isolated
overhead relative to both bundled and local execution. Every primary timer
stops when its backend returns; result destruction, collection, buffer
retirement, and allocator reset are excluded and reported as post-return
cleanup. JSON output also records nearest-rank p95 and backend-keyed 1,000-call
sequential high-frequency `add_scalar` runs, both allocating and with reusable
`out=` storage.

High-frequency call latency has the same backend-return boundary. Its
cleanup-inclusive throughput includes per-call result destruction and
reclamation when outputs are not reused. These are deliberately distinct
boundaries and neither measurement is batched.

Separate diagnostics report worker startup, RPC-only round trips, shared-memory
allocation, uncached input preparation, first-use FD passing and worker mapping,
cached mapping checks, and runtime-owned output allocation. Input preparation
includes memfd allocation, the runtime mapping and Torch view, and the ingress
copy. Ensure-mapped timings include native dispatch, FD transfer, worker
mapping, and acknowledgement. Numerical-library warmup and worker startup are
excluded from steady-state operation timings.

Every per-case diagnostic summary uses the configured diagnostic iteration
count. Uncached and cached invocation diagnostics are separate, equally sized
sample populations. Component timings are nested within and overlap their
associated invocation, so adding components does not reconstruct end-to-end
time.

Profiled invocations additionally separate scalar binding, output-plan
evaluation, C++ queue wait, RPC, worker input/output view construction, worker
dispatch, and kernel execution. The JSON report groups diagnostics by
provenance: Python frontend, RPC/control, mapping/transport, allocation,
reclamation, or kernel. Remaining Python work, including access reservation,
descriptor assembly, reusable-output validation, and result wrapping, stays in
isolated call latency but has no synthetic "bookkeeping" timer: the nested
profile components overlap, so subtracting them would not produce a valid
measurement. Profiling first showed repeated worker view construction was the
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

Use the three backends as controlled comparisons on the same machine, build,
dtype, thread count, operation shape, warmup, and iteration count. Local versus
bundled measures plugin/native-kernel integration cost, although local PyTorch
and the plugin entry point are not identical call paths. Bundled versus isolated
is the primary isolation comparison because both use the same
transport-neutral C++ kernel. Local versus isolated remains the user-facing
end-to-end comparison. Compare medians together with p95 and standard deviation,
repeat runs before attributing small differences, and use reusable `out=` runs
to separate output lifetime from mandatory eager-call control cost.

Do not move scalar binding, output planning, or remaining Python bookkeeping to
C++ merely because it is on the call path. Move one only after an opt-in profile
repeatedly identifies that named boundary as material to end-to-end latency for
a representative workload and a prototype demonstrates improvement beyond
run-to-run spread. Treat already-small components as a reason to stop: optimize
mapping, allocation/reclamation, RPC scheduling, output reuse, or kernels when
their own measurements dominate. There is intentionally no absolute or
percentage latency gate; reports inform a deployment tradeoff rather than a
pass/fail performance test.

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
all underlying options. Packaged reference benchmarks require bundled plugin
support and fail with a direct error when it is absent; source-tree smoke tests
must explicitly inject a substitute backend or skip. The output allocation
service metric measures
metadata-driven runtime allocation and output mapping before the single
operation RPC. Lazy page faults remain part of isolated end-to-end time.
The checked-in [`benchmarks/baseline.json`](benchmarks/baseline.json) and
[`benchmarks/arena.json`](benchmarks/arena.json) have the schema 9 report shape
but contain no fabricated samples until the packaged reference benchmark is
rerun. [`benchmarks/README.md`](benchmarks/README.md) links the retained schema 5
measurements and summarizes their historical primary results.
