# Contributing Documentation

WMFS uses one documentation pipeline and two source-comment conventions.

## Python

Write Google-style docstrings on public modules, classes, methods, and functions.
Describe behavior and ownership constraints rather than restating type hints.

```python
def operation(value: Tensor, *, enabled: bool = True) -> Tensor:
    """Apply the operation.

    Args:
        value: Input tensor. The function does not mutate it.
        enabled: Whether to apply the transformation.

    Returns:
        The transformed tensor.

    Raises:
        ValueError: If the input shape is unsupported.
    """
```

## C++

Write Doxygen comments on public declarations under `inc/wmfs`. Use `@brief`,
`@param`, `@return`, and `@throws` where they add information. Ownership and
thread-safety guarantees belong in the class or function comment.

```cpp
/// @brief Duplicate the descriptor with close-on-exec enabled.
/// @param minimum Lowest descriptor number accepted by `fcntl`.
/// @return An owning descriptor.
/// @throws std::system_error if duplication fails.
UniqueFd duplicate_cloexec(int minimum = 0) const;
```

## Architecture Guides

Use MyST Markdown under `docs`. Prefer `literalinclude` for implementation
snippets so examples stay synchronized with source code. Run `just doc`; Sphinx
warnings and Doxygen documentation errors fail the build.
