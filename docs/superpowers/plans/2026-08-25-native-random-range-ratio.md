<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native random range ratio implementation plan

> **For Sol:** execute each task with strict RED→GREEN evidence. Use sccache and
> `/mnt/4tb/aiperf-origin-port-056-target`; never edit the shared checkout.

**Goal:** Port upstream `94fee7338b` as a native synthetic-dataset contract with exact
vLLM/SGLang range and seeded draw-order behavior.

**Architecture:** Parse a typed ratio/style at the Config-v2 boundary, lower it into one
checked dataset policy, preseed style-specific ISL/OSL/offset vectors once, and let the
synthetic composer consume them. Inject the offset policy into the existing random prompt
factory while retaining native exact-length and raw-token paths.

**Tech stack:** Rust 2024, serde/clap, existing NumPy PCG64 compatibility code, a small
private RandomState-compatible MT19937 adapter, runtime unit tests, and mock-server E2E.

### Task 1: Preserve exact upstream ancestry

1. Commit this finding, spec, and plan.
2. Start an `ours` merge with exact second parent `94fee7338b...`.
3. Apply only `dd3f09b0c3..94fee7338b` to the merge tree; resolve target-delta context
   without importing any earlier ancestor.
4. Verify the merge's two parents and target-only tree delta, then commit.

### Task 2: Add the checked policy test-first

1. Add failing tests for scalar/object parsing, vLLM/SGLang bounds, zero/one endpoints,
   non-finite/out-of-range values, style restrictions, special-token adjustment, and
   sequence-mixture/stddev exclusivity.
2. Run the focused runtime test and capture RED.
3. Implement typed `RandomCorpusStyle`, ratio input, checked bounds, and validation.
4. Run the same test and capture GREEN; commit the focused slice.

### Task 3: Add exact preseed streams test-first

1. Add pinned upstream vectors proving all ISLs precede all OSLs and offsets for PCG64.
2. Add pinned NumPy RandomState vectors and 64-bit XOR-fold tests for SGLang.
3. Run RED.
4. Implement the private style-owned preseed plan and bounded fallback.
5. Run GREEN and commit.

### Task 4: Wire random prompt style test-first

1. Add failing generator tests for vLLM special exclusion, SGLang full vocabulary,
   offset-plus-request ordinal arithmetic, cache exhaustion, exact repair, and raw IDs.
2. Implement the injected random prompt policy without adding locks or per-token logs.
3. Prove RED→GREEN and commit.

### Task 5: Wire public configuration test-first

1. Add failing clap, YAML, Config-v2 serde, resolver, and dataset-build tests.
2. Add the two flags/fields and lower them into the checked plan.
3. Update generated CLI reference/schema only through repository generators when needed.
4. Prove RED→GREEN and commit.

### Task 6: Prove product behavior

1. Add Rust mock-server E2E for bounds, style, exact lengths, and identical seeded runs.
2. Add composer regressions for nonempty-prefix zero body and request ordinal stability.
3. Run the focused E2E and runtime engine suites, then format/clippy.
4. Commit evidence and implementation together.

### Task 7: Prove Python/native production parity

Before review, run the mandatory Python/Rust byte-exact A/B gates:

1. Generate deterministic reference cases with NumPy PCG64/default_rng and legacy
   RandomState MT19937 over multiple seeds, scalar/split/boundary ratios, special-token
   rules, and all-ISL→all-OSL→offset order.
2. Commit the Python/NumPy provenance and generated JSON fixture.
3. Consume every case from Rust, checking complete length/offset vectors and
   offset-plus-request-index token IDs.
4. Run the generator in `--check` mode and the Rust parity test as required gates.
5. Write reproducible Config-v2 cases and checked-in network-independent tokenizer
   fixtures for both zero and two automatic special tokens.
6. Launch the real Python profile and real native `aiperf profile` against one
   deterministic request-capturing Rust mock for six cases × eight requests.
7. Compare ordered method, route, content-type, exact body bytes, and re-tokenized prompt
   IDs. Do not construct the Rust payload inside the test.

### Task 8: Perform three independent semantic audits

1. Audit RNG/reference streams and boundaries, citing exact upstream lines, native seams,
   commands, and observed outcomes.
2. Separately audit config/dataset/formatter/prefix and special-token semantics.
3. Separately audit Python-to-native production E2E protocol/output behavior.
4. Enumerate every upstream runtime, typing/plumbing, test, docs/schema, and mock-recorder
   change exactly once across the audits. Record no unresolved divergence only after the
   rebuilt 13-test/48-capture production gate passes.

### Task 9: Independent review and closure

1. Request an independent Graham review over the first-parent port range.
2. For every finding, add a reproducing RED test, implement the minimal fix, and prove
   GREEN in a separate commit; request re-review until approved.
3. Record commands/results and approval in the finding and campaign row.
4. Run final verification, commit closure, and report the branch tip for a merge (never a
   cherry-pick, because the exact upstream second parent is required).
