# AGENTS.md

## Project Conventions

- The `wmfs` Python distribution is in `packages/wmfs`; its import package is
  `packages/wmfs/wmfs`.
- The independent Python plugin SDK is in `packages/wmfs-plugin`; plugins may
  depend on `wmfs_plugin` but must not depend on the main `wmfs` runtime. Only
  Python worker distributions depend on this SDK.
- Runtime protocol metadata, schemas, and ring codecs live under
  `packages/wmfs/wmfs/protocol` and `packages/wmfs/wmfs/transport`. The runtime,
  C++ builds, and `wmfs-tool` must not depend on `packages/wmfs-plugin`.
- The interface compiler and code generator are in `packages/wmfs-tool`; its
  import package and command-line entry point are `wmfs_tool` and `wmfs-tool`.
- Plugin interface definitions are the source of truth. `wmfs-tool` generates
  the deployment manifest, C++ plugin ABI headers/stubs, and Python
  metadata/stubs. Do not maintain a second handwritten registry.
- Generated C++ plugin-facing code must compile as C++11. Generated Python
  plugin-facing code must support the minimum Python 3 version declared by the
  stable plugin ABI. The main runtime and private native code use C++20 or
  newer.
- Implement the runtime in Python first.
- Use nanobind instead of pybind11 for future Python/C++ bindings.
- Keep the root `CMakeLists.txt`, C++ sources in `src`, and headers in `inc`.
- Build C++ out of tree under the ignored `build` directory.
- Commit messages must follow the rules in `.gitlint`.

## Test Layout

- Keep package unit tests with the package they test:
  `packages/wmfs/tests`, `packages/wmfs-plugin/tests`, and
  `packages/wmfs-tool/tests`.
- Keep runtime integration tests under `tests/integration` and Python
  runtime/worker-SDK interoperability tests under `tests/python_worker`.
- Keep C++ unit tests under `tests/cpp` and run them through CTest.
- Package unit tests must not require another WMFS package unless that package
  is a declared public dependency.
- Integration tests cover generated artifacts, runtime/plugin compatibility,
  process isolation, command/completion rings, shared tensors, and installed
  package behavior.

## Goal

Prototype a scientific-computing runtime where users write ordinary Python code, while selected function calls transparently execute in isolated worker processes.

The primary motivation is ABI/runtime isolation:

- main application and plugins may use different glibc versions;
- plugins must not link against the main execution framework;
- plugin implementations should be independently deployable;
- numerical data should avoid serialization/copying where practical.

The prototype must compare isolated execution against an equivalent single-process implementation and quantify the overhead.

## User Experience

User code should look ordinary:

```python
c = matmul(a, b)
u, s, vh = svd(c)
```

Users should not explicitly construct RPC requests, shared-memory handles, graphs, or worker objects.

Functions provided by plugins are dynamically registered into the Python-facing API at runtime.

Calling a registered function may actually mean:

```text
Python call
    -> runtime dispatch
    -> command ring submission
    -> isolated worker
    -> numerical kernel
    -> completion ring
    -> tensor result
```

Function registration and metadata exchange occur once during worker startup.
The hot path must not perform per-operation discovery, registration, or RPC.

Command submission and completion are internally asynchronous and support
multiple in-flight operations per worker. Preserve the existing public API:
ordinary calls such as `c = wmfs.matmul(a, b)` return tensor results and do not
expose command objects, rings, RPC, or mandatory futures. A separate explicit
async API may be added later, but is not required for the migration.

## Architecture

```text
Python Frontend
      |
      | normal function calls
      v
Execution Runtime
      |
      | versioned startup handshake
      |
      +----------------------+
      |                      |
      v                      v
Worker A                 Worker B
glibc/toolchain A        glibc/toolchain B
NumPy/PyTorch/etc.       NumPy/PyTorch/etc.
      ^                      ^
      | command/completion   | command/completion
      | shared-memory rings  | shared-memory rings
      |                      |
      +----------+-----------+
                 |
          shared CPU buffers
          memfd + mmap
```

The runtime owns object/buffer lifetime.

Workers should only receive capabilities to the inputs required for an operation and capabilities for output allocation.

Do not expose the complete object store to plugins.

## Interface Generation

Plugin authors define a versioned, declarative interface specification. At
build time, `wmfs-tool` consumes that specification and emits:

```text
plugin interface definition
          |
      wmfs-tool
          |
  +-------+----------+----------------+
  |                  |                |
manifest.json   generated C++   generated Python
                 C++11 ABI      metadata/stubs
```

The generated manifest is read by the runtime without importing plugin code.
Generated C++ and Python adapters depend only on the stable plugin ABI, not on
private runtime implementation details.

The specification and generated artifacts carry explicit format, ABI, and
feature versions. Additive evolution must preserve old generated plugins where
possible. Incompatible changes require a new ABI version and a clear startup
diagnostic; never silently reinterpret old records.

A plugin should explicitly describe:

- function name;
- input tensors;
- output tensors;
- scalar parameters;
- read-only versus mutable inputs where applicable.
- output shape and dtype expressions when they are known from inputs and scalars.

Read-only must be the default.

Mutation must be explicit.

Known outputs are preallocated by the runtime and passed to the worker in the
operation command. Use the output-allocation protocol only for genuinely
dynamic outputs whose shape cannot be declared in metadata.

## Startup And Rings

At startup the runtime:

1. Reads and validates the generated manifest.
1. Registers exported functions into the namespaced Python API once.
1. Launches the isolated worker.
1. Performs a small, versioned handshake.
1. Establishes one runtime-to-worker command ring.
1. Establishes one worker-to-runtime completion ring.
1. Establishes the FD-control channel and initial shared tensor mappings.

Cap'n Proto may remain temporarily as a migration-only startup/control
mechanism, but it is not part of the target plugin ABI or operation hot path.
The final startup handshake is owned by the stable protocol generated by
`wmfs-tool`.

Ring records use fixed-width, process-independent values. Never place raw
pointers, process-local FD numbers, C++ object layouts, or Python object details
in shared memory. Every command and completion carries a session generation,
submission ID, operation ID, bounded descriptor counts, and explicit status.

The command ring is single-producer/single-consumer for one runtime/worker
session unless a later protocol version explicitly adds another concurrency
model. The completion ring is the reverse direction. Publication and
consumption use documented atomic memory ordering. Ring-full behavior applies
backpressure rather than overwriting unread records. Blocking waits use a
kernel notification primitive such as `eventfd`; unbounded busy-spinning is not
acceptable.

Malformed records, impossible indices, generation mismatches, and ring
invariant violations are fatal session errors. Ordinary algorithm failures are
recoverable completion records and do not tear down the worker.

## Tensor Transport

Do not serialize numerical tensor payloads through the startup protocol or
command/completion rings.

For CPU tensors:

1. Runtime allocates storage using `memfd_create`.
1. Storage is mapped using `mmap`.
1. The backing FD is transferred to the worker using Unix-domain-socket `SCM_RIGHTS`.
1. The worker maps the same storage.
1. Construct NumPy/PyTorch/DLPack-compatible tensor views over the mapped memory.
1. Numerical kernels operate directly on that memory.

A ring tensor descriptor should contain metadata such as:

```text
buffer capability/id
offset
byte length
dtype
shape
strides
```

Never serialize raw pointers or process-local FD numbers.

DLPack is the numerical interoperability ABI after memory has been mapped into the receiving process. DLPack is not itself the cross-process shared-memory transport.

Cache mappings/FDS where practical. Do not repeatedly pass and map the same shared buffer for every operation.

The default allocator pools whole memfds rather than exposing one global arena.
Reuse requires the last Torch storage alias to be released, worker mappings to
be retired, and the buffer generation to advance. A trusted-plugin arena mode
may trade per-buffer capabilities for one persistent writable mapping, but it
must remain explicit and opt-in.

## Output Allocation

Output ownership must remain with the runtime.

Support an allocator capability exposed to workers.

Conceptually:

```python
def operation(ctx, a, b):
    out = ctx.empty(shape=..., dtype=..., device=...)
    kernel(a, b, out=out)
    return out
```

The worker may request output storage through the completion/control protocol,
but the runtime performs and controls the allocation. Dynamic allocation must
not grant unrestricted allocator or object-store access.

The Python-facing API may still naturally return values:

```python
c = matmul(a, b)
```

`c` is a managed tensor handle/view, not a copied control-message payload.

For operations with known output shape, allow runtime preallocation.

For operations whose output shape is not known in advance, allow the worker to request allocation dynamically.

## Algorithm Contract

Algorithm implementations should remain usable without the execution runtime.

Core numerical implementations should therefore operate on standard tensor-like values rather than object-store internals.

Preferred conceptual interface:

```text
read-only tensor inputs
+ scalar parameters
-> tensor outputs
```

For explicit in-place operations:

```text
mutable tensor input
-> modified tensor
```

Algorithms must not need to understand:

- object IDs;
- persistence;
- history/versioning;
- memfd;
- mmap;
- FD passing;
- transport details.

Runtime adapters handle those concerns.

## Python Compatibility

Prefer PyTorch as the initial tensor/numerical substrate.

The isolated and non-isolated implementations should expose the same Python API.

Example:

```python
a = torch.randn(4096, 4096)
b = torch.randn(4096, 4096)

c = matmul(a, b)
u, s, vh = svd(c)
```

The backend selection should be configurable, for example:

```python
runtime.use_backend("local")
runtime.use_backend("isolated")
```

Do not require application code to change between the two modes.

## Prototype Operations

Implement at least:

### Matrix multiplication

```python
c = matmul(a, b)
```

Use a mature numerical implementation underneath, e.g. PyTorch/BLAS.

This tests a relatively large-compute/low-control-overhead operation.

### SVD

```python
u, s, vh = svd(a)
```

Use the same underlying numerical library in both execution modes where possible.

This tests:

- multiple outputs;
- dynamic output allocation;
- more substantial computation;
- shared-memory return values.

Also implement one deliberately cheap operation, such as:

```python
b = add_scalar(a, 1.0)
```

This is important because it exposes fixed control-plane/process-isolation
overhead.

## Execution Modes

Every benchmark operation must have two implementations using equivalent numerical kernels.

### Local

```text
Python
  -> PyTorch/native function
```

Everything executes in one process.

### Isolated

```text
Python
  -> runtime
  -> command ring
  -> worker process
  -> same numerical kernel
  -> completion ring
```

Tensor payloads are shared through mapped memory.

## Benchmarking

Measure separately:

- local kernel execution time;
- isolated end-to-end execution time;
- command/completion ring round-trip latency;
- command enqueue and completion dequeue cost;
- ring-full backpressure and wakeup cost;
- first-use FD passing + mmap cost;
- repeated-call cost with mappings cached;
- shared-memory allocation cost;
- output allocation cost.

Benchmark several tensor sizes.

At minimum:

```text
small    - control-plane overhead dominates
medium   - mixed
large    - computation dominates
```

For matrix multiplication and SVD report:

```text
local time
isolated time
absolute overhead
percentage overhead
```

Warm up numerical libraries before timed measurements.

Avoid counting worker startup in steady-state benchmarks. Report startup separately.

Use multiple iterations and report median plus a spread metric such as p95 or standard deviation.

## Process Isolation Demonstration

The prototype must demonstrate that the worker can run with a runtime environment incompatible with the main process.

Ideally package:

```text
main runtime:
    newer glibc/toolchain

plugin worker:
    older/different glibc/toolchain
```

or the reverse.

The important requirement is that no plugin shared library is loaded into the main process.

Communication occurs only through the generated startup protocol,
command/completion rings, FD-control channel, and shared tensor mappings.

Nix may be used to create reproducible incompatible environments.

## Plugin Discovery

A plugin should contain:

```text
plugin interface specification
generated manifest.json
generated C++11 and/or Python adapters
worker executable/entry point
implementation
```

At startup, the execution runtime discovers available plugins and registers their exported operations.

The Python layer should then expose those operations dynamically.

Keep the first implementation simple. Explicit plugin directories or configuration are acceptable.

Do not build a general package manager yet.

## Capability Model

Workers must not receive unrestricted access to the object store.

For each command, create an operation-scoped context containing only the
required capabilities:

```text
input buffer capabilities
output allocator capability
optional device capability
logging/error reporting
submission/completion identity
```

Read-only input access should be the default.

Writable access must be explicitly declared by the operation.

This preserves dependency clarity and makes future scheduling/security work possible.

## Out of Scope for Initial Prototype

Do not implement yet:

- persistent object storage;
- automatic version history;
- copy-on-write object graphs;
- distributed/multi-host execution;
- automatic computation graph optimization;
- CUDA IPC;
- GPU scheduling;
- fault-tolerant job recovery;
- arbitrary semantic object types;
- general plugin dependency resolution.

Design interfaces so these can be added later, but do not block the prototype on them.

## Suggested Milestones

Implementation status:

- [x] Milestone 1: local PyTorch API.
- [x] Milestone 2: Cap'n Proto worker RPC and dynamic plugin registration.
- [x] Milestone 3: shared CPU tensor transport with `memfd_create`, FD passing,
  and `mmap`.
- [x] Milestone 4: isolated execution with the same Python-facing API.
- [x] Milestone 5: verified execution in a separately pinned glibc/toolchain
  environment.
- [x] Milestone 6: local-versus-isolated benchmarking.
- [x] Milestone 7: `wmfs-tool` interface compiler and stable generated plugin
  ABI.
- [x] Milestone 8: startup-only registration and versioned ring handshake.
- [x] Milestone 9: asynchronous command/completion ring hot path.
- [x] Milestone 10: dynamic allocation and recoverable errors over rings.
- [x] Milestone 11: compatibility fixtures built from older generated plugin
  artifacts.
- [x] Milestone 12: ring-versus-RPC benchmark and removal of per-call RPC.

### Milestone 1

Implement local Python API using PyTorch.

Required:

```python
matmul()
svd()
add_scalar()
```

### Milestone 2

The original prototype implemented Cap'n Proto worker RPC with ordinary
serialized scalar/control messages and dynamic plugin registration. Retain it
only as migration input while implementing Milestones 7-12.

Verify dynamic plugin registration.

### Milestone 3

Implement `memfd_create` + Unix FD passing + `mmap`.

Expose mapped memory to PyTorch/NumPy without payload serialization.

### Milestone 4

Run `matmul`, `svd`, and `add_scalar` in the isolated worker with the exact same Python-facing API.

### Milestone 5

Run the plugin worker in a deliberately different glibc/toolchain environment.

### Milestone 6

Benchmark local versus isolated execution.

Implemented by `wmfs-benchmark`, with a reproducible reference report in
`benchmarks/baseline.json`. The benchmark covers small, medium, and large cases;
reports median, p95, and standard deviation; and separates worker startup, RPC,
shared-memory transport, cached mappings, and output allocation costs.

The ring architecture adds a baseline that separates enqueue, wakeup, worker
queueing, kernel, completion, and result materialization. Keep the old RPC
baseline for an explicit before/after comparison until migration is complete.

The central success criterion is:

> For sufficiently expensive numerical operations, process isolation should add only a small fixed control-plane cost while tensor payloads remain zero-copy shared memory.

## Design Principle

Keep the boundary small.

The execution framework owns:

```text
processes
startup handshake
command/completion rings
buffer allocation
shared memory
lifetime
dispatch
```

Plugins own:

```text
algorithms
numerical kernels
local temporary variables
plugin-specific dependencies
```

The plugin ABI should essentially be:

```text
typed operation schema
+
tensor capabilities
+
small scalar metadata
```

Everything else should remain private to either side.

The stable plugin boundary is the generated ABI plus ring record format, not
the main runtime's Python package, C++ classes, Cap'n Proto version, or build
toolchain.
