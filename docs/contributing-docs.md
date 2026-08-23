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

Architecture diagrams are D2 sources under `docs/diagrams`; deterministic SVGs
under `docs/_static/diagrams` are committed so readers do not need D2. Run
`bash docs/generate-diagrams.sh` after editing a source, or
`bash docs/generate-diagrams.sh . --check` to verify freshness. The Nix development
shell supplies the pinned D2 executable, and every documentation build checks
the committed SVGs before invoking Sphinx.

GitHub Pages publishes master documentation under `latest/` and release
snapshots under `versions/<tag>/`. A master build checks the `gh-pages` branch
and backfills only missing semantic-version tags; it never rebuilds an existing
version directory. A tag push publishes that tag once. The site root is a
generated version index. Use the workflow's optional `version` input to backfill
an unpublished historical tag manually.
