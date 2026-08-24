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

## Engineering Principles

### Type Safety Where Representable

Prefer typed declarations and generated validation over convention, free-form
strings, or unchecked dictionaries whenever the value domain is known.

- The interface definition declares tensor dtype relationships, shape
  relationships, mutability, scalar types, finite option enums, output arity,
  and configuration schemas.
- Use generated enums for finite algorithm choices rather than passing string
  names through the C++ boundary.
- One operation ID may support multiple declared dtype variants. Generated
  C++11 code performs validated dtype dispatch into typed overloads or templates;
  Python stubs expose the corresponding accepted types without creating
  separate operation names per dtype.
- `wmfs-tool` rejects inconsistent defaults, examples, dtype variables, shape
  references, VJPs, and configuration schemas at build time.
- The runtime validates untrusted manifests, startup frames, ring records,
  capabilities, and configuration values again at process boundaries.
- Keep opaque bytes or free-form JSON only for genuinely open-ended data. Do not
  weaken a type that can be represented in the interface merely to simplify an
  adapter.

Type safety must not introduce C++ ABI coupling. Generated boundary records use
fixed-width, standard-layout values; typed C++ wrappers are source-level C++11
conveniences around those records.

### Optional Abstractions Need A Bypass

Every optional framework abstraction must have a supported disabled or direct
path whose nominal steady-state cost is zero, or as close to zero as the
language and ABI permit. Select the path during build or initialization rather
than repeatedly rediscovering it in the operation hot path.

Examples:

- local and bundled execution bypass worker launch, rings, FD transfer, shared
  allocation, isolated VJPs, and transport error translation;
- disabled logging creates no channel or queue, performs no serialization or
  allocation, and allows expensive message construction to be skipped with
  `enabled()`;
- disabled profiling performs no clock reads or metric record construction;
- known outputs bypass dynamic planning;
- cached mappings bypass repeated FD transfer and `mmap`;
- absent configuration is canonical `{}` and creates no per-operation work.

Do not hide unavoidable costs behind the word "zero-cost." Document and
benchmark initialization, one-time registration, indirect calls, validation,
and any remaining branch. Tests for a bypass must assert the absence of the
relevant subprocesses, FDs, mappings, allocations, serialization, locks,
syscalls, and clock reads, not only that the returned value is correct.

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

Process isolation is an execution option, not a requirement imposed on user
algorithms. Every plugin must retain a supported exit route to ordinary
single-process Torch execution and, where applicable, a unified build that
links the plugin into the main process.

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

The same registered operation catalog must be executable through three
adapters without changing application calls:

```text
                         +-> pure Torch / local
Python frontend -> API --+-> bundled / unified process
                         +-> isolated worker rings
```

Only initialization and backend selection may differ. Steady-state user code,
function signatures, tensor arguments, return values, and optional `out=` usage
must remain backend-independent.

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

`wmfs-tool` output must be execution-mode-neutral. Do not generate separate
local, bundled, and isolated interface files for one plugin. The same generated
manifest, operation IDs, metadata, declarations, and plugin entry points must
be usable in every mode. Ring-specific generation must not become the only way
to invoke an operation.

Backend selection belongs in the runtime and build system:

- an isolated worker target links the plugin implementation and generated
  entry table into a worker executable;
- a bundled target links the same implementation and generated entry table
  into the application-side extension or executable;
- a pure-Python target imports the same ordinary Torch implementation directly.

Mode-specific transport glue should be generic runtime code rather than a
second set of per-plugin generated sources. If a language binding requires thin
glue, it must consume the same generated declarations and must not redefine the
operation catalog.

The specification and generated artifacts carry explicit format, ABI, and
feature versions. Additive evolution must preserve old generated plugins where
possible. Incompatible changes require a new ABI version and a clear startup
diagnostic; never silently reinterpret old records.

A plugin should explicitly describe:

- function name;
- input tensors;
- output tensors;
- scalar parameters;
- accepted tensor dtypes and shared dtype variables;
- finite scalar options as enums where applicable;
- read-only versus mutable inputs where applicable.
- output shape and dtype expressions when they are known from inputs and scalars.
- optional initialization configuration schema, defaults, descriptions, and
  examples.

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
1. Supplies the optional initialization JSON and selected logging service.

Startup and lifecycle control use the fixed, versioned control ABI over a
private Unix `SOCK_SEQPACKET` socket. `wmfs-tool` emits the manifest identity,
interface/configuration fingerprints, lifecycle features, and startup
capabilities consumed by that handshake; ring, eventfd, log, and FD-control
descriptors are transferred with `SCM_RIGHTS`. Cap'n Proto exists only in
archived prototype history and schema 5 benchmark baselines. It is not a live
runtime dependency, startup mechanism, plugin ABI, or operation path.

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

## Worker Initialization Services

Configuration and logging are session-level host services. They are supplied
once during plugin initialization and must not become required parameters of
the numerical kernels.

The generated, execution-mode-neutral plugin entry table may expose optional
`initialize` and `shutdown` hooks. Isolated, bundled, and pure-Torch adapters
call the same logical initialization hook before publishing operations.

### Initialization Configuration

The application may provide one optional JSON object per plugin. The runtime
serializes it once using canonical UTF-8 JSON with sorted keys, compact
separators, and no NaN or infinity. The initial protocol limit is 64 KiB.

Configuration rules:

- the top-level value is an object; absence is equivalent to `{}`;
- configuration is immutable for one initialized session;
- worker replacement replays the exact canonical bytes;
- configuration travels in a bounded startup frame, not command-line
  arguments, environment variables, or per-operation records;
- the runtime validates framing and JSON syntax, while the plugin validates
  plugin-specific semantics and should version its configuration schema;
- initialization rejection is a startup error and prevents publication of the
  plugin session;
- configuration bytes and secrets must not be copied into diagnostics or logs
  by default.

When a plugin accepts configuration, its `interface.toml` declares a typed WMFS
configuration-schema v1 subset. The supported subset initially includes nested
objects, Boolean, integer, number, string, homogeneous arrays, required fields,
defaults, enums, numeric bounds, string/array length bounds, descriptions, and
`additional_properties = false`.

TOML is the authoring format for the interface and may also be used by an
application for human-written configuration. Python loads that TOML into a
normal mapping before runtime configuration. The process boundary still uses
canonical JSON bytes: do not require a TOML parser in the generated C++11 ABI.
Plugins may choose any C++11-compatible JSON parser internally, and future
generated typed accessors may hide parsing without changing the wire format.

`wmfs-tool` validates the schema, defaults, and named examples at build time and
emits them into the manifest with a configuration-schema version and
fingerprint distinct from the operation-interface fingerprint. The runtime
validates application configuration against the generated schema before worker
launch. The plugin still validates semantic constraints during initialization.

Configuration introspection is manifest-driven and does not require launching
or importing the worker:

```python
wmfs.list_configurable()
wmfs.list_configurable("reference")
wmfs.runtime.validate_config("reference", config)
```

The returned metadata includes field types, requirements, defaults, bounds,
enum values, descriptions, and validated examples. The worker startup handshake
advertises only the configuration-schema version and fingerprint and confirms
acceptance of the supplied configuration; it does not resend the complete
schema on every startup.

Python initialization receives the decoded object and a logger:

```python
def initialize(config: dict[str, object], logger: Logger) -> None:
    ...
```

C++11 initialization receives borrowed process-local views:

```cpp
status initialize(json_view config, logger log);
```

`json_view` contains `const char*` plus an explicit fixed-width byte length. The
pointer is local to the process and valid only for the initialization call. It
is never placed in shared memory. The plugin may parse or copy the bytes using
its own C++ library; no `std::string` crosses the generated interface.

Local and bundled initialization receive the same logical configuration
without creating a worker or transport. Pure Python receives a normal decoded
mapping; bundled C++ receives the same canonical JSON view.

### Logging

Python and C++ plugins use matching logger semantics:

- levels `debug`, `info`, `warning`, `error`, and `critical`, with numeric values
  10, 20, 30, 40, and 50;
- `enabled(level)` for avoiding expensive message construction;
- `log(level, message, category, fields)` plus level-specific convenience
  methods;
- structured fields limited initially to Boolean, signed/unsigned 64-bit
  integer, float64, and bounded UTF-8 text values;
- contextual child loggers that bind plugin, worker, session, operation,
  submission, and invocation identity;
- thread-safe, non-throwing calls whose failure never changes operation status.

The C++11 interface uses borrowed `text_view` values, fixed-width field records,
and a process-local function table/context pointer. It must not expose
`std::string`, virtual classes, exceptions, or STL object layouts across the
plugin boundary. The Python SDK exposes an equivalent `Logger` protocol and may
adapt it to the standard `logging` package.

Logging mode is selected by the host for each worker session:

- **centralized:** a worker-local logger sends bounded structured records over
  a dedicated nonblocking Unix `SOCK_SEQPACKET` channel to a runtime collector,
  which emits ordinary Python `logging.LogRecord` values;
- **disabled:** the worker receives a null logger, `enabled()` always returns
  false, log methods return immediately, and no log socket or serialization is
  created;
- **worker file:** the worker writes its own structured log file through a
  bounded local queue and does not send records to the main process.

Do not reuse command, completion, or FD-control channels for logs. Logging must
not delay completion delivery or create transport deadlocks. Centralized and
file sinks use bounded queues. On saturation, they may drop records according
to configured severity, count drops, and emit one synthetic warning when
delivery resumes. Oversized values are truncated with explicit flags.

The runtime may also capture worker stdout and stderr as an unstructured
fallback for third-party libraries, but explicit structured logging is the
preferred path. Error completions remain separate from logs.

Tests must verify initialization runs exactly once per session, canonical
configuration parity across execution modes, exact replay after worker
replacement, startup rejection, logger level filtering, contextual fields,
drop accounting, worker-file output, and centralized collection. Disabled-mode
tests must prove that no log channel is created and repeated disabled log calls
perform no formatting, allocation, serialization, or transport work at the
worker boundary.

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

For isolated execution, output ownership must remain with the runtime. Local
and bundled execution use normal Torch allocation and `out=` semantics.

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

## Unified And Pure-Torch Exit Route

Supporting degeneration to ordinary Torch is a hard requirement.

Plugin numerical implementations operate on ordinary Torch tensors or
tensor-like values and scalar parameters. They must remain callable without:

- worker processes;
- command or completion rings;
- shared-memory descriptors;
- `memfd`, `mmap`, or FD passing;
- runtime object IDs or capability records;
- an `InvocationContext` in the core numerical kernel.

Generated or handwritten transport adapters may translate an
`InvocationContext` or ring record into a call to that kernel, but transport
concerns must stop at the adapter boundary.

The execution modes are:

- **Pure Torch/local:** call the ordinary Python Torch implementation in the
  application process using native Torch allocation, views, exceptions, and
  autograd.
- **Bundled/unified:** call the same plugin entry points and implementation from
  a directly linked or in-process build using native tensors. This mode bypasses
  worker startup, fixed control channels, rings, and shared-memory transport.
- **Isolated:** use the generated metadata, runtime-owned shared storage, and
  command/completion rings around the same mathematical kernel.

Initialization for local or bundled execution must be small and bounded by the
number of plugins and operations. It may read manifests, validate generated
metadata, build dispatch tables, and import the selected in-process
implementation once. It must not launch workers, establish rings, map shared
arenas, perform per-operation discovery, or allocate tensors proportional to
user data. Measure and report initialization separately from steady-state
calls.

After initialization, local and bundled calls should be indistinguishable from
regular Torch calls except for the dynamic `wmfs` function lookup:

- inputs and outputs are ordinary Torch tensors;
- output allocation follows Torch conventions;
- views and `out=` preserve normal Torch behavior;
- native Torch autograd is preferred over isolated VJP machinery;
- algorithm exceptions propagate as ordinary local exceptions;
- no mandatory future, command, graph, or runtime tensor wrapper is exposed.

Every public reference operation must have contract tests across all available
local, bundled, and isolated modes. Tests cover signatures, values, dtypes,
shapes, multiple outputs, `out=`, views, errors, and first-order autograd where
the operation supports it. A feature is incomplete if it works only through
the isolated transport.

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
runtime.use_backend("bundled")
runtime.use_backend("isolated")
```

Do not require application code to change between execution modes.

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

Use the same underlying numerical library in all execution modes where
possible.

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

Every benchmark operation must be available through local, bundled when built,
and isolated adapters using equivalent numerical kernels.

### Local

```text
Python
  -> PyTorch/native function
```

Everything executes in one process.

This is the pure-Torch exit route. It uses ordinary tensors and Torch semantics
and must not initialize isolated transport resources.

### Bundled / Unified

```text
Python
  -> generic in-process runtime adapter
  -> same generated plugin entry table
  -> same directly linked plugin kernel
  -> native Torch tensor result
```

Everything executes in the application process. This mode trades away ABI and
crash isolation but preserves the same public operation API.

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

- local, bundled, and isolated initialization time;
- local kernel execution time;
- bundled end-to-end execution time;
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
bundled time
isolated time
absolute overhead
percentage overhead
```

Warm up numerical libraries before timed measurements.

Avoid counting worker startup in steady-state benchmarks. Report startup separately.

Use multiple iterations and report median plus a spread metric such as p95 or standard deviation.

Save retained benchmark reports with explicit comparison names that identify
the measured revision or role, such as `benchmark-0.1.0.json`,
`benchmark-master.json`, and `benchmark-master-arena.json`. Do not overwrite or
reuse an ambiguous baseline name when results from multiple revisions are being
compared.

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
- [x] Milestone 13: verify every plugin interface and implementation can be
  built for local, bundled, and isolated execution without mode-specific
  generated interface files.
- [x] Milestone 14: implement mode-neutral initialization hooks, typed
  configuration schemas and introspection, canonical JSON configuration, and
  optional centralized, null, and worker-file loggers for both Python and C++
  plugins.

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
`benchmarks/benchmark-master.json`. The benchmark covers small, medium, and
large cases; reports median, p95, and standard deviation; and separates
initialization, fixed startup control,
shared-memory transport, cached mappings, and output allocation costs.

The ring architecture adds a baseline that separates enqueue, wakeup, worker
queueing, kernel, completion, and result materialization. Historical reports
must retain their original field names and measurement boundaries; current
reports must not relabel old samples.

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

The stable plugin boundary is the generated ABI plus fixed control and ring
record formats, not the main runtime's Python package, private C++ classes, or
build toolchain.
