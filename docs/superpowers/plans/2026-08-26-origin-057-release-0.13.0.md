# Origin #57 release 0.13.0 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:test-driven-development for every behavior slice and
> superpowers:verification-before-completion before each success claim.

**Goal:** Restore upstream `0.13.0` shared-product release metadata and
documentation while preserving the independent native Rust version boundary.

**Architecture:** Apply the four-line upstream metadata/documentation delta
unchanged. Record the disposition and merge it with a dedicated closure branch;
do not modify Cargo packages or invent a native test for Python packaging.

**Tech Stack:** Hatchling/Python package metadata, Markdown documentation, Git
ancestry verification.

**Spec:** `docs/specs/2026-08-26-origin-057-release-0.13.0-boundary.md`

## Global Constraints

- Base is exactly `9a13a98884219487972b3c8295a3d6bfebffcc4a`.
- Preserve exact upstream release values from `e10d53b1d3`; do not cherry-pick.
- Do not alter Cargo manifests, native Rust behavior, or unrelated package data.
- Use `RUSTC_WRAPPER=sccache`, `SCCACHE_DIR=/mnt/4tb/sccache-port057`, and
  `CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port057` if a Rust verification
  command is required.

---

### Task 1: Release metadata restoration

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/aiperf_mock_server/pyproject.toml`
- Modify: `docs/plugins/creating-your-first-plugin.md`
- Modify: `docs/server-metrics/server-metrics-json-schema.md`

- [x] Write and run an exact-surface assertion that fails while `0.13.0` is
  absent.
- [x] Apply only the four-file upstream release delta.
- [x] Run the exact-surface assertion and inspect the four-file diff.
- [x] Commit the upstream-equivalent closure.

### Task 2: Release boundary and integration evidence

**Files:**
- Create: `artifacts/archives/origin-main-findings/commit-057-e10d53b1d3.md`
- Create: `docs/specs/2026-08-26-origin-057-release-0.13.0-boundary.md`
- Create: `docs/superpowers/plans/2026-08-26-origin-057-release-0.13.0.md`
- Modify: `docs/porting-origin-main-campaign.md`

- [x] Record the Python-package/native-Rust boundary and no-native-test ruling.
- [x] Add Graham review evidence.
- [x] Commit the implementation closure, await the root-provided current head,
  and create the final two-parent integration merge.
