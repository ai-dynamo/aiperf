# Stock evaluator environments

These two projects lock the mutually independent Python dependency universes
used to generate and deploy AIPerf's stock evaluator root filesystems. They are
not dependencies of the AIPerf frontend or Rust workspace.

Create a provider root with the pinned CPython 3.12 runtime and the committed
lock, for example:

```text
uv sync --frozen --project tools/stock_evaluators/nemo --python 3.12.10
uv sync --frozen --project tools/stock_evaluators/openbench --python 3.12.10
```

The stock-manifest generator requires both resulting virtual-environment
prefixes explicitly. It verifies every distribution version and `RECORD`
before using either environment; the ambient AIPerf `.venv` is never a source.
