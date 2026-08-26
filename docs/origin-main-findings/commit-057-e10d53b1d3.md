# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Origin #57 findings: release version 0.13.0

Upstream `e10d53b1d30b5845f56cbea63d0560f10ff5aa4e` is a coordinated
shared-product release action. It advances the Python distribution and Python
mock-server package metadata from `0.12.0` to `0.13.0`, and updates the two
user-facing documentation examples that show the emitted package or metrics
version. It has no Rust source, Cargo manifest, or Rust test changes.

## Ancestry and tree state

The exact upstream object is already an ancestor of the shared base
`9a13a98884219487972b3c8295a3d6bfebffcc4a`. Its semantic tree delta was not
retained: all four upstream-owned version surfaces still displayed `0.12.0`.
This closure restores exactly those four lines of release metadata and examples:

- root `pyproject.toml` package version;
- `tests/aiperf_mock_server/pyproject.toml` package version;
- the installed-package example in the plugin tutorial; and
- the three `aiperf_version` examples in the server-metrics schema guide.

The Rust workspace has independent Cargo package versioning. No Cargo manifest,
native CLI artifact contract, or runtime behavior is changed by this release
metadata action.

## Test disposition

Upstream supplies no test. The release contract is static package/documentation
metadata, so the focused verification is an exact-surface assertion over the
four upstream files. Adding a Rust test would test unrelated packaging tooling
and would not make the Python distribution metadata more observable to the
native runtime.

## Verification inventory

- `git merge-base --is-ancestor e10d53b1d3 9a13a9888421` succeeds.
- The pre-change exact-surface assertion fails because every upstream-owned
  surface reports `0.12.0`.
- The post-change exact-surface assertion requires `0.13.0` in both package
  manifests and each updated documentation example.

## Graham review outcome

The implementer's root-independent Graham review made two passes over the exact
metadata delta and the final merged closure and found no findings. The changed
production surface is metadata only; no hot path, async, allocation,
synchronization, tracing, or Rust API behavior is in scope. The diff remains
limited to the four upstream release surfaces and the required closure records.
