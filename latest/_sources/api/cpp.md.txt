# C++ API Reference

C++ documentation is generated from Doxygen comments in `inc/wmfs` and rendered
inside this Sphinx site through Breathe.

## Native Control Session

```{doxygenenum} wmfs::native::TensorDType
```

```{doxygenenum} wmfs::native::ScalarKind
```

```{doxygenclass} wmfs::native::Session
---
members:
---
```

```{doxygenstruct} wmfs::native::Mapping
---
members:
---
```

```{doxygenstruct} wmfs::native::TensorDescriptor
---
members:
---
```

```{doxygenstruct} wmfs::native::ScalarArgument
---
members:
---
```

```{doxygenstruct} wmfs::native::InvocationProfile
---
members:
---
```

## Worker Mapping API

```{doxygenstruct} wmfs::reference::MappingSpec
---
members:
---
```

```{doxygenclass} wmfs::reference::MappedBufferCache
---
members:
---
```

```{doxygenclass} wmfs::reference::TensorLease
---
members:
---
```

```{doxygenfunction} wmfs::reference::receive_buffer_transfers
```

## Reference Kernels

```{doxygenfunction} wmfs::reference::matmul
```

```{doxygenfunction} wmfs::reference::matmul_out
```

```{doxygenfunction} wmfs::reference::svd
```

```{doxygenfunction} wmfs::reference::svd_out
```

```{doxygenfunction} wmfs::reference::add_scalar
```

```{doxygenfunction} wmfs::reference::add_scalar_out
```

```{doxygenfunction} wmfs::reference::matmul_vjp_out
```

```{doxygenfunction} wmfs::reference::add_scalar_vjp_out
```

## File Descriptor Ownership

```{doxygenclass} wmfs::UniqueFd
---
members:
---
```
