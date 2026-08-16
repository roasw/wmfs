# WMFS Documentation

WMFS executes ordinary Python tensor calls locally, through an in-process
bundled plugin, or in an isolated worker process. The isolated path uses Cap'n
Proto for control messages and shared `memfd` mappings for tensor payloads.

```{toctree}
---
maxdepth: 2
---
getting-started
data-flow
api/python
api/cpp
contributing-docs
```

## Documentation Model

This is one Sphinx site for both implementation languages:

- Python API pages are generated from Google-style docstrings by `autodoc`.
- C++ API pages are generated from Doxygen comments by Doxygen and Breathe.
- Architecture and reading guides are MyST Markdown with source-backed
  `literalinclude` snippets.

Run `just doc` to build the complete site reproducibly. Open
`build/Debug/docs/html/index.html` after the default build completes.
