# C++ API Reference

C++ documentation is generated from Doxygen comments in `inc/wmfs` and rendered
inside this Sphinx site through Breathe.

## Native Control Session

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

## Worker Mapping API

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
