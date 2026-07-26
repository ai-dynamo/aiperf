# Rust Comment and Provenance Cleanup Design

## Goal

Review every hand-written Rust source file and make its comments, rustdoc, and
non-preserved naming match the Graham review style: concise, current, and
limited to information that cannot be read directly from the code.

## Scope

The review covers every `.rs` file under:

- `rust/runtime`
- `rust/cli`
- `rust/mock-server`
- `rust/e2e-tests`
- `rust/loadgen-core`
- `rust/pyext`
- Rust build scripts and Rust tools in the workspace

The current workspace contains roughly 600 Rust files, 240,000 lines, and more
than 34,000 comment lines. Every in-scope file must appear in the execution
manifest and receive an explicit reviewed status.

Generated data, golden files, embedded corpora, fixture payloads, and
vendor-shaped protocol or constants modules are excluded from prose rewrites.
They remain in the manifest with an explicit exclusion reason. A required code
rename may update a fixture reference, but must not rewrite fixture content for
style.

## Preservation Policy

The cleanup preserves:

- The required two-line NVIDIA SPDX and Apache-2.0 license header.
- CLI command names and flag names.
- YAML identifiers.
- Citations required to define a wire, schema, standard, algorithm, or
  byte-exact compatibility contract.
- Comments that explain safety, invariants, cancellation, concurrency,
  scheduling, timing, allocation, wire behavior, serialization, or another
  non-obvious interaction.
- Public rustdoc that explains semantics a caller cannot infer from the item
  name and type signature.

Existing public CLI and YAML names remain unchanged even if they contain
otherwise prohibited wording. The implementation may describe those names
without repeating their historical framing.

## Removal and Rewrite Policy

Prohibited historical framing includes provenance, migration, porting,
prior-layout, supersession, stage or track, and refactor narratives. The scan
matches grammatical variants and equivalent phrases, not only a fixed word
list. Domain uses remain valid when they do not express history: a TCP port,
portable code, and a refactored mathematical quantity are not provenance.
Identifiers containing `legacy` are prohibited except for preserved CLI and
YAML names.

Remove comments and rustdoc that:

- Record implementation provenance or prior source locations.
- Describe migrations, ports, former layouts, stages, tracks, slices, or
  refactors.
- Narrate the following statement or restate a function, field, test, or type
  name.
- Preserve development history that belongs in version control.
- Repeat an invariant already documented at the owning abstraction.
- Use proof, step, phase, or section banners without adding a constraint.
- Document private items whose behavior is obvious from their name and type.

Retained prose states the current contract directly. It does not compare the
implementation with an earlier implementation unless the comparison itself is
an externally observable compatibility requirement. Contract-critical
citations remain, but implementation-history attribution does not.

Except for preserved CLI commands, CLI flags, and YAML identifiers, prohibited
wording is also removed from internal identifiers and strings. Renames must be
complete across definitions, references, tests, logs, and artifacts. External
wire names containing prohibited historical framing must also change, and each
such change must be called out in the final report.

## Comment Style

Comments must be direct and technically specific:

- Prefer one sentence over a paragraph.
- Explain why a constraint exists, not what the next line does.
- Put public API semantics in `///` and internal implementation constraints in
  `//`.
- Keep `SAFETY` comments immediately attached to the unsafe operation they
  justify.
- Keep protocol field numbers, serialization edge cases, clock requirements,
  and cancellation or backpressure invariants where they are enforced.
- Remove stale crate names and source-path citations unless the path is the
  contract source.
- Do not add comments merely to replace deleted comments.

## Parallel Ownership Model

Twenty-four agents edit disjoint file sets in the same worktree. Before
dispatch, the controller generates an exact manifest assigning every in-scope
Rust file to one agent or to an excluded-file list. No file may have more than
one owner.

The ownership groups are:

1. Runtime engine bootstrap, application, coordinator, and execution core.
2. Runtime scheduled, sharded, worker, and turn execution.
3. Runtime engine cellular controller, cell, aggregator, and launch paths.
4. Runtime engine graph, offline, Dynamo, and dry-run execution.
5. Runtime HTTP transport.
6. Runtime gRPC and transport-core code.
7. Runtime graph compilation, execution, recorded inputs, and stores.
8. Runtime datasets, body plans, and content serving.
9. Runtime endpoints and extension registries.
10. Runtime clocks, timing, workloads, schedulers, and phase orchestration.
11. Runtime metrics, reports, and exporters.
12. Runtime accuracy, adaptive control, telemetry, cellular support, and
    remaining top-level modules.
13. CLI process entry, dispatch, execution modes, signals, and logging.
14. CLI configuration models.
15. CLI loading, YAML, expansion, flags, and profile projection.
16. CLI sweep, search, fitting, and optimization.
17. CLI synthesis, standalone commands, resources, build script, and tests.
18. Mock-server request handling, generation, latency, scheduling, and state.
19. Mock-server gRPC, listeners, configuration, balancing, and TLS.
20. Mock-server metrics, telemetry, accuracy, tests, examples, and tools.
21. E2E harness, raw-record helpers, cellular, graph, and fold tests.
22. E2E endpoint, gRPC, Riva, media, and tool-call tests.
23. E2E infrastructure, export, telemetry, accuracy, search, and remaining
    tests.
24. `loadgen-core`, `pyext`, and remaining Rust build or tool sources.

The generated manifest resolves overlaps and files that do not fit these
descriptions before agents start. Large files remain single-owner; agents do
not split one file by line range.

## Agent Contract

Each agent must:

1. Read every assigned file completely.
2. Classify each comment as preserve, tighten, delete, or excluded.
3. Apply the preservation and removal policies without unrelated code changes.
4. Rename prohibited internal identifiers and strings completely within its
   ownership set.
5. Record cross-partition references that the controller must reconcile.
6. Report files inspected, files changed, files intentionally untouched,
   exclusions, external wire changes, and checks run.

Agents must not run workspace-wide formatters or modify files outside their
manifest. Partition-local checks may run concurrently only when they do not
write shared generated state.

## Dirty-Tree Safety

The worktree already contains extensive user changes. Before editing, the
controller records:

- `git status --short`
- The tracked diff, including staged changes
- The untracked-file list
- A file manifest with hashes for all in-scope Rust files

The cleanup must edit current file contents in place and must not restore,
discard, stage, or commit existing work. Final review compares the resulting
tree with this baseline and attributes only the new comment and rename edits.

## Integration

After all agents finish, the controller:

1. Verifies that every manifest entry has a terminal status.
2. Reconciles cross-partition renames and stale references.
3. Runs one formatting pass.
4. Scans all Rust sources for prohibited wording.
5. Reviews every scan exception against the CLI, YAML, SPDX, exclusion, and
   contract-citation policies.
6. Inspects the combined diff for behavior changes, fixture churn, deleted
   invariants, and unrelated edits.
7. Runs focused tests for renamed interfaces and each affected crate.
8. Runs workspace-level checks proportional to available feature dependencies
   and reports any check that cannot run.

No commit is created unless the user explicitly requests one.

## Verification

Minimum completion gates:

- Every in-scope `.rs` file is reviewed and represented in the manifest.
- No prohibited wording remains outside approved exceptions.
- CLI command names, flag names, and YAML identifiers are unchanged.
- SPDX headers remain present.
- Generated, vendor-shaped, golden, corpus, and fixture content has no
  style-only churn.
- `cargo fmt --check` passes.
- Rustdoc links compile for affected crates.
- `cargo check` passes for affected workspace members and available feature
  sets.
- Focused crate and integration tests pass for renamed code and externally
  visible changes.
- Final cross-partition review finds no stale references or lost safety,
  protocol, concurrency, timing, cancellation, or serialization rationale.

## Deliverable

The final report lists:

- Files reviewed, changed, and excluded.
- Comment lines removed or rewritten.
- Identifiers and strings renamed.
- Preserved CLI and YAML exceptions.
- External wire or artifact changes.
- Verification commands and results.
- Any unresolved dependency, feature, or environment limitation.
