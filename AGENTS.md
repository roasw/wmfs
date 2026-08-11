# AGENTS.md

## Goal

The project is called `wmfs`, and the Python module should placed in `wmfs` in project root.

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
    -> Cap'n Proto RPC
    -> isolated worker
    -> numerical kernel
    -> RPC result
```

## Architecture

```text
Python Frontend
      |
      | normal function calls
      v
Execution Runtime
      |
      | Cap'n Proto RPC/control
      |
      +----------------------+
      |                      |
      v                      v
Worker A                 Worker B
glibc/toolchain A        glibc/toolchain B
NumPy/PyTorch/etc.       NumPy/PyTorch/etc.
      |                      |
      +----------+-----------+
                 |
          shared CPU buffers
          memfd + mmap
```

The runtime owns object/buffer lifetime.

Workers should only receive capabilities to the inputs required for an operation and capabilities for output allocation.

Do not expose the complete object store to plugins.

## RPC

Use Cap'n Proto for:

- plugin interface definitions;
- operation metadata;
- RPC request/response;
- function discovery/registration;
- tensor metadata;
- errors/status.

Prefer generating plugin registration code from the Cap'n Proto interface/schema rather than maintaining a second handwritten function registry.

A plugin should explicitly describe:

- function name;
- input tensors;
- output tensors;
- scalar parameters;
- read-only versus mutable inputs where applicable.

Read-only must be the default.

Mutation must be explicit.

## Tensor Transport

Do not serialize numerical tensor payloads through Cap'n Proto.

For CPU tensors:

1. Runtime allocates storage using `memfd_create`.
1. Storage is mapped using `mmap`.
1. The backing FD is transferred to the worker using Unix-domain-socket `SCM_RIGHTS`.
1. The worker maps the same storage.
1. Construct NumPy/PyTorch/DLPack-compatible tensor views over the mapped memory.
1. Numerical kernels operate directly on that memory.

A serialized tensor descriptor should contain metadata such as:

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

The worker may request output storage, but the runtime performs/controls the allocation.

The Python-facing API may still naturally return values:

```python
c = matmul(a, b)
```

`c` is a managed tensor handle/view, not a copied RPC payload.

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
- RPC details.

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

This is important because it exposes the fixed RPC/process-isolation overhead.

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
  -> Cap'n Proto RPC
  -> worker process
  -> same numerical kernel
```

Tensor payloads are shared through mapped memory.

## Benchmarking

Measure separately:

- local kernel execution time;
- isolated end-to-end execution time;
- RPC-only round-trip latency;
- first-use FD passing + mmap cost;
- repeated-call cost with mappings cached;
- shared-memory allocation cost;
- output allocation cost.

Benchmark several tensor sizes.

At minimum:

```text
small    - RPC overhead dominates
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

Communication must occur only through the defined RPC/shared-memory boundary.

Nix may be used to create reproducible incompatible environments.

## Plugin Discovery

A plugin should contain:

```text
plugin manifest/schema
generated Cap'n Proto bindings
worker executable/entry point
implementation
```

At startup, the execution runtime discovers available plugins and registers their exported operations.

The Python layer should then expose those operations dynamically.

Keep the first implementation simple. Explicit plugin directories or configuration are acceptable.

Do not build a general package manager yet.

## Capability Model

Workers must not receive unrestricted access to the object store.

For each invocation, create an operation-scoped context containing only the required capabilities:

```text
input buffer capabilities
output allocator capability
optional device capability
logging/error reporting
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

### Milestone 1

Implement local Python API using PyTorch.

Required:

```python
matmul()
svd()
add_scalar()
```

### Milestone 2

Implement Cap'n Proto worker RPC with ordinary serialized scalar/control messages.

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

The central success criterion is:

> For sufficiently expensive numerical operations, process isolation should add only a small fixed control-plane cost while tensor payloads remain zero-copy shared memory.

## Design Principle

Keep the boundary small.

The execution framework owns:

```text
processes
RPC
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
