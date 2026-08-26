# Native aiohttp minimum closure

## Decision

Do not add a Rust compatibility layer for upstream
`9e96b499d1e942e1b338010fa5d99d9d9c96e074`. Its entire product contract is
the Python package resolver range `aiohttp>=3.14.3,<4`, which is already in
the shared package manifest through an exact upstream merge.

## Native boundary

The native executable uses the Rust Hyper transport. Aiohttp is a Python-only
dependency and is absent from all native Cargo manifests. Consequently, no
native API, wire request, response reduction rule, configuration field, or
integration test can observe this dependency floor.

## Acceptance evidence

The exact upstream commit is an ancestor of the campaign head, current
`pyproject.toml` contains the exact range, and all `rust/*/Cargo.toml` manifests
are free of aiohttp. This is sufficient evidence of complete shared-product
coverage and native non-applicability.
