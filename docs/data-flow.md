# Data Flow And Reading Guide

This guide follows the `matmul(a, b)` call in the introductory script through
the native isolated path. The Python-control worker follows the same planning
and memory model, but performs Cap'n Proto calls with pycapnp instead of the
native session extension.

## End-To-End Flow

```text
user script
  -> wmfs.api.matmul
  -> Runtime.invoke
  -> IsolatedBackend.invoke
  -> bind_invocation / BufferManager
  -> NativeWorkerSession or WorkerSession
  -> batched SCM_RIGHTS mapping control
  -> Cap'n Proto invokeKnown RPC
  -> ReferenceServer::run_known
  -> MappedBufferCache tensor views
  -> generated operation adapter
  -> reference numerical kernel
  -> shared output tensor returned to Python
```

### 1. Public Call

`wmfs.api.matmul` deliberately contains no transport concepts. It forwards the
ordinary Python arguments to the selected runtime.

```{literalinclude} ../packages/wmfs/wmfs/api.py
---
language: python
pyobject: matmul
---
```

Read next:

- `packages/wmfs/wmfs/api.py`: stable user-facing functions.
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
  autograd edge and backward RPC.
- `packages/wmfs-plugin/wmfs_plugin/metadata.py`: canonical operation and VJP
  declarations, validation, and fingerprints.

### 3. Invocation Planning

The transport-neutral planner binds tensor and scalar arguments, applies access
metadata, computes output shape/dtype plans, and validates reusable `out=`
tensors. Python and native control paths consume the same plan.

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

Unmanaged CPU tensors are copied once into runtime-owned contiguous storage.
`BufferManager.empty` allocates either a pooled `memfd` region or an arena
subrange, then creates a Torch view. Known outputs are allocated before
dispatch.

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

### 5. Mapping And RPC Dispatch

The selected session batches all required mappings. Cap'n Proto carries buffer
IDs, generations, tensor metadata, operation IDs, and scalar values. Tensor
payload bytes never enter the RPC message.

Important implementations:

- `packages/wmfs/wmfs/transport/native_worker.py`, `NativeWorkerSession`:
  Python orchestration around the nanobind native control path.
- `src/native_session.cpp`, `Session::map_buffers` and `Session::invoke`:
  synchronous KJ RPC and batched `SCM_RIGHTS` control.
- `packages/wmfs/wmfs/transport/worker_process.py`, `WorkerSession`: equivalent
  Python/pycapnp control path.
- `packages/wmfs/wmfs/transport/fd_broker.py`, `FdSender.ensure_mapped_many`:
  Python batched FD sender.
- `packages/wmfs-plugin/wmfs_plugin/schemas/wmfs/runtime.capnp`: operation RPC
  and metadata protocol.
- `packages/wmfs-plugin/wmfs_plugin/schemas/wmfs/tensor.capnp`: tensor and
  batched buffer-transfer descriptors.

### 6. Worker Views And Kernel Dispatch

The C++ worker receives FDs on its control socket and caches mappings by buffer
generation. ATen storage captures shared mapped-region ownership, so retained
tensor aliases remain valid after cache retirement. The RPC thread constructs
views, invokes a generated transport adapter, and calls handwritten numerical
kernels.

Read in this order:

1. `src/reference_worker.cpp`, `ReferenceServer::run_known`: RPC request scope,
   view construction, profiling, and dispatch.
1. `src/reference_mapped_buffers.cpp`, `MappedBufferCache::tensor`: descriptor
   validation and zero-copy ATen views.
1. `plugins/reference/generated/reference_dispatch.inc`: generated operation ID
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
RPC response contains only completion status and optional metrics. Python
returns the preallocated managed Torch tensor. Releasing its last storage alias
queues the allocation for later collection. Collection retires worker mappings
in batches. In pooled mode it resets the whole region, advances its generation,
and places it in the size-matched pool; in arena mode it returns and coalesces
the allocation's subrange without recycling the arena mapping.

## Local And Bundled Differences

- `LocalBackend.invoke` calls PyTorch directly and bypasses shared memory and
  RPC.
- `BundledBackend.invoke` calls the same C++ reference kernels in process through
  `wmfs._bundled`, bypassing shared memory and RPC.
- Comparing isolated against bundled in `wmfs-benchmark` most directly measures
  process-isolation overhead for the same native kernels.
