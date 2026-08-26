# Origin #53 Enum Inequality Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` task-by-task. This closure is executed in an isolated worktree; no change may be made directly to the shared branch.

**Goal:** Restore the upstream Python case-insensitive enum `!=` contract and prove the native equivalent already has the invariant.

**Architecture:** Python's `str` MRO requires explicit `__ne__` forwarding, whereas Rust canonicalizes identifier text once and derives equality on stored canonical bytes. The implementation restores Python behavior only; Rust contributes focused proof, not a duplicate abstraction.

**Tech Stack:** Python `Enum`/pytest; Rust derive macros, Cargo, sccache.

**Spec:** `docs/specs/2026-08-26-origin-053-enum-inequality-restoration.md`

## Global Constraints

- Preserve `NotImplemented` so reflected Python comparisons remain available.
- Use the existing `/mnt/4tb` target directory and `RUSTC_WRAPPER=/usr/bin/sccache` for Cargo tests.
- Do not change native public interfaces merely to mimic Python inheritance.
- The exact upstream object is already an ancestor; the restoration commit documents the tree-level regression instead of manufacturing a duplicate merge.

---

### Task 1: Restore and prove normalized Python inequality

**Files:**
- Modify: `src/aiperf/common/enums/base_enums.py`
- Modify: `src/aiperf/plugin/extensible_enums.py`
- Modify: `tests/unit/common/enums/test_base_enums.py`
- Modify: `tests/unit/plugin/test_extensible_enums.py`

**Interfaces:**
- Produces: `__ne__(self, other) -> bool | NotImplemented` on both Python enum bases.
- Consumes: the existing normalized `__eq__` implementations.

- [x] **Step 1: Write the failing upstream regression matrix.**

  Add assertions that normalized case/separator matches have `!= is False`,
  each `!=` result is the negation of `==`, registered extensions retain the
  rule, and direct unsupported operands return `NotImplemented`.

- [x] **Step 2: Run RED verification.**

  Run: `PYTHONPATH=/mnt/4tb/aiperf-origin-port-053/src /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/pytest -q tests/unit/common/enums/test_base_enums.py tests/unit/plugin/test_extensible_enums.py`

  Expected: normalized spellings fail because inherited `str.__ne__` compares
  raw bytes. Observed: 32 failures.

- [x] **Step 3: Implement the smallest restoration.**

  Add `result = self.__eq__(other)` to each base; return it unchanged when it
  is `NotImplemented`, otherwise return `not result`. Restore the upstream
  exception-based lazy cache reads and registered-extension cache test.

- [x] **Step 4: Run GREEN verification.**

  Run the same two pytest modules. Expected: all tests pass.

### Task 2: Prove native equivalence without adding a false abstraction

**Files:**
- Inspect: `rust/runtime/src/extensions/registry_id.rs`
- Inspect: `rust/runtime/src/engine/protocol_v2.rs`

**Interfaces:**
- Consumes: `RegistryId::new` and `ComponentId::from_str` normalization.
- Produces: evidence that derived Rust inequality is the negation of equality.

- [x] **Step 1: Inspect construction and derive boundaries.**

  Confirm both types store canonicalized strings and derive `Eq` plus
  `PartialEq`, rather than offering Python-style raw-vs-normalized comparison.

- [x] **Step 2: Run focused native evidence.**

  Run: `RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port053 cargo test -p aiperf-runtime extensions::registry_id --lib`.
  Source-audit `ComponentId`, whose historical test module is intentionally
  disabled by `#[cfg(any())]`; do not claim its selector as runtime evidence.

- [x] **Step 3: Record the no-Rust-change ruling.**

  Do not add a wrapper or manual `PartialEq` implementation: derived equality
  makes an independent, inconsistent `!=` path impossible.

### Task 3: Close the source-regression record

**Files:**
- Create: `docs/origin-main-findings/commit-053-e5ebe915df.md`
- Create: `docs/specs/2026-08-26-origin-053-enum-inequality-restoration.md`
- Create: `docs/plans/2026-08-26-origin-053-enum-inequality-restoration.md`
- Modify: `docs/porting-origin-main-campaign.md`
- Modify: `docs/specs/README.md`

- [x] **Step 1: Record ancestry-vs-tree evidence.**

  State that the upstream object is an ancestor but its semantic tree delta was
  absent, so a restoration commit is necessary and another actual merge cannot
  be constructed.

- [x] **Step 2: Review the full closure diff and commit it.**

  Run the focused verification, formatter/checks proportional to the modified
  Python modules, then record self-Graham findings before committing outside
  the sandbox.
