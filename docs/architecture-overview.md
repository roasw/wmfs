# Architecture Overview

WMFS lets application code make ordinary synchronous Python tensor calls while
the runtime may execute the kernel in a persistent, ABI-isolated worker. The
runtime owns discovery, planning, process lifecycle, rings, capabilities, and
shared CPU storage. Plugins own typed interfaces and numerical kernels.

The diagrams use both shape and labels, not color alone. Blue rectangles are
application-runtime components; orange hexagons are isolated workers; purple
documents are generated or declarative artifacts; teal parallelograms are
transport channels; green cylinders are shared memory; gold circles are
compute; red double-bordered boxes are fatal session failures; and amber diamonds are
recoverable operation failures. Dashed lines denote build-time, optional, or
explicitly non-hot-path relationships.

## Recommended Reading Order

1. **Public API:** read [Getting Started](getting-started.md), then the hot-path
   diagram below to establish that calls are synchronous while dispatch is
   internally asynchronous.
1. **Packages and generation:** read the package-role and interface-generation
   diagrams, then [Package Roles](package-roles.md) and
   [Plugin Artifact Compatibility](plugin-compatibility.md).
1. **Discovery and startup:** read the startup diagram before tracing
   `Runtime.discover_plugins` and session construction.
1. **Rings:** read the hot-path diagram, then the normative
   [Command and Completion Ring Protocol](ring-protocol.md).
1. **Buffers and FDs:** read the tensor-transport diagram before the detailed
   [Data Flow](data-flow.md) guide.
1. **Workers and outputs:** compare worker boundaries, then known and dynamic
   output allocation.
1. **Failures:** read failure/recovery before changing transport validation,
   deadlines, or worker lifecycle.
1. **Evidence:** finish with `wmfs.benchmark`, `benchmarks/README.md`, and the
   integration, Python-worker interoperability, and C++ protocol tests linked
   below.

## 1. Package Roles

```{figure} _static/diagrams/package-roles.svg
---
alt: Runtime, build, Python worker, and C++ worker package roles.
width: 100%
---
WMFS distributions belong to separate build and runtime environments; they are
not a dependency stack installed everywhere.
```

The application installs `wmfs`. Only a Python worker installs `wmfs-plugin`.
Plugin build and CI environments run `wmfs-tool`, then package its deterministic
outputs with a worker. The runtime does not import plugin implementations, and a
C++ worker has no dependency on either Python package used by plugin authors.

**Key invariants and nuances**

- `wmfs`, `wmfs-plugin`, and `wmfs-tool` have deliberately one-way, disjoint
  dependency roles.
- The Python SDK is only for Python workers. It is not an application dependency
  and is not used by C++ workers.
- Runtime and worker environments may use different Python, Torch, libc, and
  toolchain versions.

**Read the implementation**

- [`pyproject.toml`](../pyproject.toml): application runtime package configuration
  at the repository root.
- [`packages/wmfs-plugin/pyproject.toml`](../packages/wmfs-plugin/pyproject.toml):
  independent worker SDK dependencies.
- [`packages/wmfs-tool/pyproject.toml`](../packages/wmfs-tool/pyproject.toml):
  independent interface compiler dependencies.
- [`nix/packages.nix`](../nix/packages.nix): separately deployable package graph.

## 2. Interface Generation

```{figure} _static/diagrams/interface-generation.svg
---
alt: Build-time interface generation and generated artifact consumers.
width: 100%
---
One declarative plugin interface drives runtime metadata and plugin-facing
artifacts.
```

`interface.toml` declares operation IDs, parameters, access, output expressions,
and VJPs. `wmfs-tool` validates it and emits committed deployment artifacts. The
runtime reads the manifest without running the tool or importing plugin code.

**Key invariants and nuances**

- The interface definition is the only handwritten operation registry.
- Generated artifacts have explicit format, ABI, protocol, generator, and
  fingerprint fields; old supported artifacts remain deployment inputs.
- The generated C ABI is future/plugin-facing. The current C++ reference worker
  uses the generated stable plugin entry table for compatibility RPC methods; its normal
  ring dispatch remains worker-specific and is not the same boundary.

**Read the implementation**

- [`plugins/reference/interface.toml`](../plugins/reference/interface.toml):
  reference plugin source of truth.
- [`wmfs_tool/generator.py`](../packages/wmfs-tool/wmfs_tool/generator.py):
  deterministic output set, fingerprints, ABI, and adapters.
- [`plugins/reference/generated/manifest.json`](../plugins/reference/generated/manifest.json):
  runtime deployment metadata.
- [`plugins/reference/generated/src/reference_plugin_stub.cpp`](../plugins/reference/generated/src/reference_plugin_stub.cpp):
  generated compatibility RPC dispatch.
- [`tests/integration/test_generated_v1_compatibility.py`](../tests/integration/test_generated_v1_compatibility.py):
  old generated artifact compatibility.

## 3. Eager Startup

```{figure} _static/diagrams/startup.svg
---
alt: Eager plugin discovery and worker startup channels.
width: 100%
---
Discovery validates and starts every candidate worker before publishing a new
operation catalog.
```

Discovery is eager and transactional: one persistent worker per plugin starts
during `discover_plugins`, not on the first operation. The fixed control ABI
carries startup identity and environment, lifecycle control, and exact ring/FD
descriptor roles. Operation traffic uses the rings after startup.

**Key invariants and nuances**

- A candidate registry is published only after all workers validate.
- Failed discovery closes all sessions created by that attempt and leaves the
  prior registry intact.
- Startup establishes two SPSC rings, four eventfds, an FD-control socket, a
  session generation, and bounded deadlines.

**Read the implementation**

- [`Runtime.discover_plugins`](../packages/wmfs/wmfs/runtime.py): candidate
  discovery, publication, and previous-backend cleanup.
- [`IsolatedBackend.discover`](../packages/wmfs/wmfs/backends/isolated.py): eager
  session creation per manifest.
- [`WorkerSession._serve`](../packages/wmfs/wmfs/transport/worker_process.py):
  Python control-mode startup and validation.
- [`NativeWorkerSession.__init__`](../packages/wmfs/wmfs/transport/native_worker.py):
  native control-mode startup orchestrated from Python.
- [`control.py`](../packages/wmfs/wmfs/protocol/control.py): fixed startup,
  lifecycle, and FD-control codec.

## 4. Steady-State Hot Path

```{figure} _static/diagrams/hot-path.svg
---
alt: Synchronous API over asynchronous command and completion rings.
width: 100%
---
The public call is synchronous; private producer and consumer machinery permits
multiple submissions to be in flight.
```

The caller blocks for its tensor result, but `_RingClient` assigns submission
IDs, queues commands, and independently consumes completions. The command and
completion rings run in opposite directions. Numerical bytes remain in shared
pages; records contain bounded fixed-width descriptors and scalar metadata.

**Key invariants and nuances**

- No public future, command, ring, worker, or shared-memory handle leaks into the
  ordinary API.
- Discovery, registration, and worker launch happen before this path. Control
  messages are not used for normal operation calls.
- Full rings apply bounded backpressure and eventfd waits; they never overwrite
  unread records or busy-spin without bound.

**Read the implementation**

- [`wmfs.__getattr__`](../packages/wmfs/wmfs/__init__.py): generation-bound
  dynamic operation publication.
- [`Runtime.invoke_registered`](../packages/wmfs/wmfs/runtime.py): synchronous
  public dispatch and lifecycle accounting.
- [`bind_invocation`](../packages/wmfs/wmfs/invocation.py): transport-neutral
  argument, access, and output planning.
- [`_RingClient`](../packages/wmfs/wmfs/transport/worker_process.py): producer,
  consumer, pending waiters, deadlines, and profiling boundaries.
- [`src/ring.cpp`](../src/ring.cpp): native SPSC publication and eventfd waits.

## 5. Tensor and FD Transport

```{figure} _static/diagrams/tensor-transport.svg
---
alt: Shared tensor memory, FD transfer, mapping, and retirement.
width: 100%
---
FD transfer establishes mappings; ring descriptors select validated tensor views
over the same CPU pages.
```

The runtime creates and owns `memfd` storage. It sends duplicate descriptors in
batched `SCM_RIGHTS` control messages only when a worker lacks the required
mapping or needs a writable upgrade. Every operation still carries capability
and view metadata in its ring record. A managed tensor can cross repeated calls
without copying; an unmanaged CPU tensor incurs one ingress copy.

**Key invariants and nuances**

- Numerical payload bytes never enter control frames or ring records.
- Read-only input access is the default; writable mappings and access leases are
  explicit. Pooled mode preserves per-buffer capabilities, while trusted arena
  mode deliberately exposes one persistent writable mapping.
- Reuse waits for the last Torch storage alias and worker retirement, then
  advances the pooled buffer generation. DLPack is not currently used and is
  not the cross-process transport.

**Read the implementation**

- [`BufferManager`](../packages/wmfs/wmfs/memory/buffers.py): allocation,
  access leases, aliases, pooling, arena mode, and reclamation.
- [`FdSender`](../packages/wmfs/wmfs/transport/fd_broker.py): runtime-side
  batched map and retirement protocol.
- [`FdReceiver`](../packages/wmfs-plugin/wmfs_plugin/fd_transport.py): Python
  worker mapped-buffer endpoint.
- [`MappedBufferCache`](../src/reference_mapped_buffers.cpp): C++ worker mapping
  validation, ATen view construction, and lifetime.
- [`control.h`](../inc/wmfs/protocol/control.h): fixed transactional mapping
  descriptors and acknowledgements, not tensor payload serialization.

## 6. Output Allocation

```{figure} _static/diagrams/output-allocation.svg
---
alt: Known and dynamic runtime-owned output allocation.
width: 100%
---
Known outputs use one ring command; genuinely dynamic outputs use two ring
commands while ownership remains with the runtime.
```

For known shape and dtype expressions, the runtime evaluates metadata and
preallocates outputs before `COMMAND_INVOKE`. Dynamic operations first issue
`COMMAND_PLAN_OUTPUTS`; the worker returns only indexed shape and dtype metadata.
After validation, the runtime allocates and maps storage, then sends the ordinary
invoke command with writable descriptors.

**Key invariants and nuances**

- Dynamic allocation means two ring commands, not worker-owned allocation and
  not a control RPC.
- The worker receives only operation-scoped input and output capabilities; it
  does not receive a general allocator or object store.
- Invalid dynamic plans are fatal transport failures because trusting them would
  violate runtime ownership and descriptor bounds.

**Read the implementation**

- [`output_metadata.py`](../packages/wmfs/wmfs/output_metadata.py): known shape
  and dtype expression evaluation.
- [`plan_outputs`](../packages/wmfs/wmfs/invocation.py): runtime validation and
  reusable `out=` handling.
- [`NativeWorkerSession._plan_dynamic_outputs`](../packages/wmfs/wmfs/transport/native_worker.py):
  native-control dynamic planning command.
- [`WorkerSession._plan_dynamic_outputs`](../packages/wmfs/wmfs/transport/worker_process.py):
  Python-control equivalent.
- [`run_ring`](../src/reference_worker.cpp): C++ planning and invocation command
  handling.

## 7. Failure and Recovery

```{figure} _static/diagrams/failure-recovery.svg
---
alt: Recoverable operation errors and fatal session recovery.
width: 100%
---
Algorithm failures are operation results; corruption and transport failures are
session failures.
```

A supported `STATUS_OPERATION_ERROR` becomes `OperationError` and leaves the
session usable. Malformed records, impossible sequence state, generation or
submission mismatches, worker exit, deadline expiry, and FD-control failures
invalidate the transport. Pending callers fail, the backend evicts and closes
the session, and a later invocation may create a fresh validated worker.

**Key invariants and nuances**

- Readers never skip malformed records to attempt resynchronization.
- Access leases and invocation-scoped writable mappings are released or
  invalidated on every path; failed pooled retirements quarantine regions.
- Shutdown is bounded: graceful close is followed by terminate/kill deadlines,
  and repeated close calls are safe.

**Read the implementation**

- [`transport/errors.py`](../packages/wmfs/wmfs/transport/errors.py): operation
  versus worker-transport exception types.
- [`_RingClient._fail`](../packages/wmfs/wmfs/transport/worker_process.py): one
  fatal error propagated to all pending submissions.
- [`IsolatedBackend._evict_session`](../packages/wmfs/wmfs/backends/isolated.py):
  failed-session eviction and cleanup.
- [`BufferManager._release_pooled_many`](../packages/wmfs/wmfs/memory/buffers.py):
  worker retirement and quarantine after failure.
- [`test_failure_boundaries.py`](../tests/python_worker/test_failure_boundaries.py):
  recoverable errors, hostile workers, deadlines, FD failures, and reaping.

## 8. Python and C++ Worker Boundaries

```{figure} _static/diagrams/worker-boundaries.svg
---
alt: Python and C++ worker dependency and transport boundaries.
width: 100%
---
Python and C++ workers share wire behavior but intentionally have different
packaging and adapter boundaries.
```

The application runtime remains Python-owned in both control modes. Native mode
uses the nanobind `wmfs._native.Session` for startup and FD control, but ordinary
ring submissions still flow through Python's `_RingClient`. A Python worker uses
the standalone SDK and generated Python adapter. The current C++ reference
worker owns a worker-specific ring loop and generated dispatch include before
calling transport-neutral LibTorch kernels.

**Key invariants and nuances**

- `wmfs-plugin` never imports `wmfs`, and the runtime never imports the SDK.
- Their independently packaged control codecs, metadata models, ring constants,
  and codecs are parity-tested.
- The generated plugin-facing C ABI describes the intended stable C++ plugin
  boundary, but current reference ring dispatch remains worker-specific.

**Read the implementation**

- [`native_worker.py`](../packages/wmfs/wmfs/transport/native_worker.py): Python
  orchestration combining native startup/FD control with `_RingClient`.
- [`src/native_session.cpp`](../src/native_session.cpp): nanobind session's
  fixed lifecycle and FD-control implementation.
- [`wmfs_plugin/worker.py`](../packages/wmfs-plugin/wmfs_plugin/worker.py): Python
  worker bootstrap and ring dispatch.
- [`src/reference_worker.cpp`](../src/reference_worker.cpp): C++ reference
  startup server and ring consumer.
- [`src/reference_kernels.cpp`](../src/reference_kernels.cpp): transport-neutral
  native numerical kernels.
- [`test_protocol_parity.py`](../tests/python_worker/test_protocol_parity.py):
  independent runtime/SDK protocol-copy parity.

## Benchmarks and Cross-Boundary Tests

The architecture is measured rather than inferred. `wmfs-benchmark` separates
worker startup, fixed startup/control ping, ring enqueue and wakeups,
backpressure, first-use FD transfer and mapping, cached mappings, output
allocation, worker view construction, kernel time, result materialization, and
reclamation. It compares local, bundled, and isolated execution using equivalent
kernels where possible.

**Read the implementation**

- [`benchmark.py`](../packages/wmfs/wmfs/benchmark.py): measurement boundaries,
  report schema, and local/bundled/isolated comparisons.
- {download}`benchmarks/README.md <../benchmarks/README.md>`: reproducible
  benchmark workflow and checked reports.
- [`tests/integration/test_benchmark.py`](../tests/integration/test_benchmark.py):
  required measurement groups and report validation.
- [`tests/integration/test_tensor_transport.py`](../tests/integration/test_tensor_transport.py):
  process-level shared tensor behavior.
- [`tests/integration/test_native_session.py`](../tests/integration/test_native_session.py):
  native control boundary.
- [`tests/cpp/ring_protocol_test.cpp`](../tests/cpp/ring_protocol_test.cpp) and
  [`tests/cpp/ring_test.cpp`](../tests/cpp/ring_test.cpp): native ABI and SPSC
  invariants.
