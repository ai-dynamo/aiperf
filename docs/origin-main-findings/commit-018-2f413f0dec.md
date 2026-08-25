# Origin #18 closure: minified generated schemas

Upstream commit `2f413f0dec` changes the Python schema generators to emit
single-line JSON using `orjson`, and adds a regression ensuring a bulk
`LoopScheduler` cancellation does not spuriously invoke the drain observer.
The generated config and plugin schemas are shipped product artifacts.

The upstream commit has one focused Python unit test and no integration or
end-to-end tests. The schema generators expose `--check` validation, which is
the applicable artifact-level regression. Native Rust does not consume these
Python-generated JSON schemas and has no scheduler drain-observer seam, so
there is no native behavior or Rust E2E port to implement.

Plan/disposition:

- preserve the exact upstream merge and generated artifact changes;
- run the focused scheduler unit test and both generator checks;
- record this as `shared-product-action`, with no Rust implementation change.

The config schema had a generated-file conflict because current shared HEAD
contains newer schema content. It was regenerated with the upstream minifying
generator after the merge, preserving the current schema semantics in the
minified artifact while retaining all other upstream #18 changes.
