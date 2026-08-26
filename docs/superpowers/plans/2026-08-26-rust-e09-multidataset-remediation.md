# E09: Reject lossy multi-dataset YAML Implementation Plan

> **Author:** Sol

**Goal:** Make unsupported multi-dataset YAML fail at the native adapter
boundary instead of silently lowering only its first entry.

**Architecture:** Keep the runtime's one-dataset `Inputs` and factory contract
unchanged.  Resolve the mutually exclusive YAML authoring forms into one
validated `DatasetSection` in `Benchmark::into_inputs`, before the existing
prompt extraction and all dataset-specific field projection.  This makes the
lossy choice impossible rather than attempting to detect it after projection.

**Spec:** `docs/specs/2026-08-26-rust-e09-multidataset-remediation.md`

## Constraints

- Do not introduce partial multi-dataset composition.
- Do not make shorthand win over an expanded list, or vice versa.
- Preserve the existing emitted inputs for each valid single-dataset form.
- Use focused Rust tests with a dedicated `/mnt/4tb` Cargo target, then obtain
  an independent Graham review before merge.

## Steps

- [ ] **1. Pin the current bad behavior with public resolver tests.** Add YAML
  fixtures for `datasets: []`, two expanded entries with distinct prompt sizes,
  and simultaneous `dataset:` plus `datasets:`.  On the unmodified adapter,
  prove at least the two-entry form resolves while retaining the first entry;
  also pin the existing omitted-dataset synthetic default as a non-regression.

- [ ] **2. Add one effective-dataset resolver at the lowering boundary.**
  Replace the direct shorthand-or-`next()` expression with a local helper or
  match that accepts a shorthand or one expanded entry, rejects empty/multiple
  expanded lists and both forms, and preserves omitted-form synthetic
  defaulting.  Return the stable exactly-one diagnostic before any authored
  dataset fields are inspected.

- [ ] **3. Turn the regression suite GREEN.** Verify all invalid authoring
  forms fail with the contract diagnostic and each valid form retains its
  expected synthetic fields.  Run the focused YAML test module in its isolated
  `/mnt/4tb` target and record the exact command/output.

- [ ] **4. Validate integration discipline.** Run formatting/diff checks and
  the appropriate broader config test scope.  Obtain a separate agent's full
  Graham review of implementation and tests; resolve every finding before
  committing the focused fix.
