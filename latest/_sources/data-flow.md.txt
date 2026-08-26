# Data Flow And Reading Guide

This guide follows the `matmul(a, b)` call in the introductory script through
the native isolated path. The Python-control worker follows the same planning,
ring, and memory model with a Python ring dispatcher.

Start with the {doc}`architecture-overview` for the system-wide diagrams; this
page then provides the detailed source-level call trace.

## End-To-End Flow

```text
user script
  -> wmfs.__getattr__("matmul")
  -> generation-bound operation callable
  -> Runtime.invoke
  -> IsolatedBackend.invoke
  -> bind_invocation / BufferManager
  -> WorkerSession with native transport
  -> batched SCM_RIGHTS mapping control
  -> command ring publication + eventfd
  -> reference worker ring consumer
  -> MappedBufferCache tensor views
  -> generated operation adapter
  -> reference numerical kernel
  -> shared output tensor returned to Python
```

### 1. Public Call

`wmfs.__getattr__` publishes only operations present in the live runtime
catalog. The returned function records the catalog generation and forwards
ordinary Python arguments to the selected runtime.

```{literalinclude} ../packages/wmfs/wmfs/__init__.py
---
language: python
pyobject: __getattr__
---
```

Read next:

- `packages/wmfs/wmfs/__init__.py`: dynamic operation publication.
- `packages/wmfs/wmfs/api.py`: static backend-aware tensor constructors.
- `packages/wmfs/wmfs/runtime.py`, `Runtime.invoke`: lifecycle-safe backend
  selection and accepted-work accounting.

### 2. Backend And Autograd Routing

`IsolatedBackend.invoke` resolves the operation's plugin metadata. Calls that
need gradients are wrapped by the generic VJP bridge; other calls dispatch
directly to the retained plugin session.

```{literalinclude} ../packages/wmfs/wmfs/backends/isolated.py
---
language: python
pyobject: IsolatedBackend.invoke
---
```

Read next:

- `packages/wmfs/wmfs/backends/isolated.py`: plugin/session ownership and
  concurrent close behavior.
- `packages/wmfs/wmfs/autograd.py`, `invoke_with_vjp`: the custom PyTorch
  autograd edge and backward ring invocation.
- `packages/wmfs/wmfs/protocol/metadata.py`: runtime operation and VJP
  declarations, validation, and fingerprints.

### 3. Invocation Planning

The transport-neutral planner binds tensor and scalar arguments, applies access
metadata, computes output shape/dtype plans, and validates reusable `out=`
tensors. Bundled and isolated paths consume the same plan.

```{literalinclude} ../packages/wmfs/wmfs/invocation.py
---
language: python
pyobject: bind_invocation
---
```

Read next:

- `packages/wmfs/wmfs/invocation.py`: binding, access reservation, ingress
  sharing, output planning, and common metrics.
- `packages/wmfs/wmfs/output_metadata.py`: evaluates schema-derived output
  shape and dtype expressions.

### 4. Runtime-Owned Shared Memory

WMFS tensor constructors and known outputs call `BufferManager.empty` to
allocate either a pooled `memfd` region or an arena subrange, then create a Torch
view directly over that storage. Unmanaged CPU tensors from ordinary Torch
constructors are copied once into the same runtime-owned contiguous storage.

```{literalinclude} ../packages/wmfs/wmfs/memory/buffers.py
---
language: python
pyobject: BufferManager.from_tensor
---
```

Read next:

- `packages/wmfs/wmfs/memory/buffers.py`, `BufferManager.empty`: allocation,
  storage leases, generations, and pool ownership.
- `BufferManager.reserve_access`: read/write scheduling for aliases.
- `BufferManager.collect`: grouped worker retirement, generation advancement,
  pooling, and reclamation instrumentation.

### 5. Mapping And Ring Dispatch

The selected session batches all required mappings over the FD-control socket,
then publishes a fixed-width command containing capabilities, tensor metadata,
operation ID, and scalar values. Tensor payload bytes never enter a ring or
control frame. The fixed control ABI handles startup identity/environment,
liveness, shutdown, and transactional FD batches.

Important implementations:

- `packages/wmfs/wmfs/transport/worker_process.py`, `WorkerSession`: startup and
  process orchestration around the mandatory native transport client.
- `src/native_session.cpp`: native command publication, completion dispatch,
  concurrent submission matching, and benchmark timing boundaries.
- `src/ring.cpp`: native SPSC publication, backpressure, and eventfd waits.
- `src/native_session.cpp`: native batched FD sender and completion dispatcher.
- `packages/wmfs/wmfs/protocol/control.py`: startup, lifecycle, and batched
  buffer-transfer protocol.
- `inc/wmfs/protocol/control.h`: language-neutral fixed control ABI.

### 6. Worker Views And Kernel Dispatch

The C++ worker receives FDs on its control socket and caches mappings by buffer
generation. ATen storage captures shared mapped-region ownership, so retained
tensor aliases remain valid after cache retirement. The ring worker constructs
views, invokes a generated transport adapter, and calls handwritten numerical
kernels.

Read in this order:

1. `src/reference_worker.cpp`, `run_ring`: command scope, view construction,
   profiling, completion, and dispatch.
1. `src/reference_mapped_buffers.cpp`, `MappedBufferCache::tensor`: descriptor
   validation and zero-copy ATen views.
1. `plugins/reference/generated/src/reference_plugin_stub.cpp`: generated operation ID
   and scalar adaptation.
1. `src/reference_kernels.cpp`: transport-independent numerical kernels.
1. `inc/wmfs/reference/kernels.hpp`: documented native kernel API.

The Python worker equivalents are:

- `packages/wmfs-plugin/wmfs_plugin/worker.py`, `_invoke_known`.
- `packages/wmfs-plugin/wmfs_plugin/invocation.py`, `InvocationContext`.
- `plugins/reference/wmfs_reference/_generated.py`: generated adapters.
- `plugins/reference/wmfs_reference/kernels.py`: ordinary Torch kernels.

### 7. Return And Reclamation

The worker writes directly into output mappings allocated by the runtime. The
completion record contains only status and optional metrics. Python
returns the preallocated managed Torch tensor. Releasing its last storage alias
queues the allocation for later collection. Collection retires worker mappings
in batches. In pooled mode it resets the whole region, advances its generation,
and places it in the size-matched pool; in arena mode it returns and coalesces
the allocation's subrange without recycling the arena mapping.

## Bundled And Isolated Differences

- `BundledBackend.invoke` calls either a Python SDK provider or a native
  `wmfs._bundled` provider in process, bypassing shared memory and rings.
- Comparing isolated against bundled in `wmfs-benchmark` most directly measures
  process-isolation overhead for the same native kernels.

Both modes consume the same generated operation catalog and lifecycle contract.
Bundled initialization receives the same logical canonical
configuration and logger service but create no worker, rings, shared allocator,
FD channel, or mappings. Disabled logging selects a null logger during
initialization, so calls perform no socket, queue, serialization, or clock work.
