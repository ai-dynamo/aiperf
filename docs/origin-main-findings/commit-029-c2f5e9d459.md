# Origin commit c2f5e9d459: cache normalized string-enum values and hashes

## Finding and audit

The upstream change is a Python-only hot-path optimization. `CaseInsensitiveStrEnum`
now caches its normalized value and hash, while `ExtensibleStrEnum` lazily caches
the same values because extension members can be registered dynamically. The
upstream unit audit covers the base enum and extensible enum implementations,
including eager/lazy cache population, identity and exact-match paths,
cross-enum equality, hash consistency, and cache fallback behavior. The commit
adds no integration or E2E test files.

The native Rust workspace has no implementation of these Python enum classes or
their normalization/hash protocol. The optimization therefore has no native
behavior to port and requires no Rust test addition. The exact non-fast-forward
merge is retained for provenance.

## Closure and review

The upstream unit tests remain available in their original locations. Native
applicability was checked against the Rust workspace; no matching enum cache
seam exists. Documentation validation and whitespace/static checks pass.
Graham review found no findings: this closure introduces no Rust diff, async or
concurrency behavior, allocation change, or error-handling surface.
