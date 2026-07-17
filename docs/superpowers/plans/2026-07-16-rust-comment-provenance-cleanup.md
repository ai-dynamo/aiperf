# Rust Comment and Provenance Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` to implement this plan task-by-task. This plan
> has a user-approved exception for one parallel wave of 24 implementers with
> disjoint file ownership; use `superpowers:dispatching-parallel-agents` for
> that wave. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Review every hand-written Rust source file, retain only concise
current-contract comments, remove historical provenance, and remove prohibited
historical naming outside preserved CLI commands, flags, and YAML identifiers.

**Architecture:** A controller captures the dirty-tree baseline and generates
an exhaustive one-owner manifest. Twenty-four agents read and edit disjoint
file sets concurrently. The controller then reconciles cross-partition
references, runs semantic and lexical scans, verifies the combined tree, and
performs a final Graham-style review.

**Tech Stack:** Rust 2024 workspace, Cargo, rustfmt, rustdoc, ripgrep, Git,
Cursor parallel subagents.

## Global Constraints

- Follow
  `docs/superpowers/specs/2026-07-16-rust-comment-provenance-cleanup-design.md`.
- Preserve the two-line NVIDIA SPDX and Apache-2.0 header in every Rust source
  file.
- Preserve CLI command names, CLI flag names, and YAML identifiers.
- Preserve only citations required to define current wire, schema, standard,
  algorithm, or byte-exact contracts.
- Remove implementation-history provenance, migration, porting, prior-layout,
  supersession, stage, track, slice, and refactor narratives.
- Retained comments explain only non-obvious safety, invariants, cancellation,
  concurrency, scheduling, timing, allocation, wire behavior, serialization,
  or interactions.
- Remove `legacy` from identifiers and strings except preserved CLI and YAML
  names.
- A network port, portable code, or a refactored mathematical quantity is not
  historical framing and must not be changed merely because of a lexical
  match.
- Do not edit generated data, golden files, embedded corpora, fixture payloads,
  or vendor-shaped protocol/constants modules for style.
- Do not restore, discard, stage, commit, or overwrite pre-existing worktree
  changes.
- No two parallel agents may own the same file.
- Do not run workspace-wide formatters from parallel agents.
- No commit is created unless the user explicitly requests one.

---

### Task 1: Capture the Dirty-Tree Baseline

**Files:**
- Read: all tracked and untracked files reported by Git.
- Create outside the repository: `/tmp/aiperf-comment-cleanup-<timestamp>/`
- Create outside the repository:
  `/tmp/aiperf-comment-cleanup-<timestamp>/status.txt`
- Create outside the repository:
  `/tmp/aiperf-comment-cleanup-<timestamp>/tracked.patch`
- Create outside the repository:
  `/tmp/aiperf-comment-cleanup-<timestamp>/staged.patch`
- Create outside the repository:
  `/tmp/aiperf-comment-cleanup-<timestamp>/untracked.txt`
- Create outside the repository:
  `/tmp/aiperf-comment-cleanup-<timestamp>/rust-hashes.txt`

**Interfaces:**
- Produces: immutable baseline directory path recorded in the progress ledger.
- Consumes: current worktree only; performs no repository writes.

- [ ] **Step 1: Record Git state**

Run from the repository root:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
baseline="/tmp/aiperf-comment-cleanup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$baseline"
mkdir -p "$baseline/controller"
git status --short > "$baseline/status.txt"
git diff --binary > "$baseline/tracked.patch"
git diff --cached --binary > "$baseline/staged.patch"
git ls-files --others --exclude-standard > "$baseline/untracked.txt"
printf '%s\n' "$baseline" > /tmp/aiperf-comment-cleanup-current
printf '%s\n' "$baseline"
```

Expected: the command prints one new baseline directory and does not change
`git status`.

- [ ] **Step 2: Hash all current Rust files**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
baseline="$(< /tmp/aiperf-comment-cleanup-current)"
git ls-files -co --exclude-standard -- '*.rs' \
  | sort \
  | xargs -r sha256sum \
  > "$baseline/rust-hashes.txt"
```

Expected: one hash entry for every tracked or untracked `.rs` file.

- [ ] **Step 3: Record baseline location durably**

The absolute baseline path was written to
`/tmp/aiperf-comment-cleanup-current` in Step 1. Read it back and verify it
points to the created directory:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
baseline="$(< /tmp/aiperf-comment-cleanup-current)"
test -d "$baseline"
printf '%s\n' "$baseline"
```

Expected: the same absolute path printed in Step 1.

- [ ] **Step 4: Verify baseline completeness**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
baseline="$(< /tmp/aiperf-comment-cleanup-current)"
test -s "$baseline/status.txt"
test -f "$baseline/tracked.patch"
test -f "$baseline/staged.patch"
test -f "$baseline/untracked.txt"
test -s "$baseline/rust-hashes.txt"
```

Expected: exit code 0.

### Task 2: Generate the Exhaustive Ownership Manifest

**Files:**
- Read: every `.rs` file under `rust/` plus workspace Rust build/tool sources.
- Create outside the repository:
  `$baseline/controller/manifest.json`
- Create outside the repository:
  `$baseline/controller/agent-01.txt` through
  `$baseline/controller/agent-24.txt`
- Create outside the repository:
  `$baseline/controller/excluded.txt`

**Interfaces:**
- Produces: exact one-owner file lists consumed by all 24 agents.
- Consumes: Task 1 baseline and the current filesystem.

- [ ] **Step 1: Inventory canonical Rust files**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
baseline="$(< /tmp/aiperf-comment-cleanup-current)"
git ls-files -co --exclude-standard -- 'rust/**/*.rs' \
  | sort -u \
  > "$baseline/controller/all-rust-files.txt"
wc -l "$baseline/controller/all-rust-files.txt"
```

Expected at plan creation: 600 files under the six canonical directories
`rust/runtime`, `rust/cli`, `rust/mock-server`, `rust/e2e`,
`rust/loadgen-core`, and `rust/pyext`. If the count differs at execution time,
reconcile the difference against Git status and the six on-disk workspace
directories before dispatch. Do not include stale editor-index paths that do
not exist on disk.

- [ ] **Step 2: Assign each file exactly once**

Apply these ordered ownership rules:

1. Runtime engine bootstrap/core: application, coordinator, execute,
   readiness, registry, protocol, control-plane, record-lane, sidecar, and
   execution-factory files.
2. Runtime scheduled execution: scheduled, sharded, turn, worker,
   request-rate, user-centric, fixed-schedule, and phase execution files.
3. Runtime engine cellular: controller, cell, aggregator, launcher,
   heartbeat, distribution identity, artifact shipping, and cellular-kind
   files under the engine or current equivalent directory.
4. Runtime engine graph/offline: graph execution/input/phase, offline,
   Dynamo, dry-run, and gRPC execution files under the engine or current
   equivalent directory.
5. Runtime HTTP transport: all current HTTP transport/client/SSE/model/config
   modules and their focused tests.
6. Runtime gRPC/core transport: gRPC transport, codec, binding, reduction,
   measurement, and transport-core modules and their focused tests.
7. Runtime graph: all `src/graph/**`, excluding data-like embedded corpora
   assigned to the exclusion list.
8. Runtime dataset/content: dataset, body-plan, content-server, and
   materialization modules and their focused tests.
9. Runtime endpoints/extensions: endpoint and extension registry modules and
   their focused tests.
10. Runtime timing/workloads: clock, timing, scheduler, workload,
    phase-runtime, scheduled top-level modules, and focused simulation tests.
11. Runtime metrics/export: metrics, metrics-core, report, export,
    server-metrics, and network-latency modules and focused tests.
12. Remaining runtime: accuracy, adaptive, failure, telemetry, cellular
    support, RNG, top-level modules, examples, benches, and unassigned runtime
    tests.
13. CLI entry/execution: main, lib, dispatch, delegate, execute modes,
    exec-bin, signals, logging, render, redaction, and cellular roles.
14. CLI models: all `rust/cli/src/model/**`.
15. CLI config/profile: config, load, YAML, expansion, flags, profile, and
    public catalog/template glue.
16. CLI sweep/search: sweep, search, fitting, optimization, isotonic, Bayes,
    and history modules.
17. Remaining CLI: synthesis, standalone commands, build script, and all CLI
    tests.
18. Mock-server request core: handlers, models, tokens, latency, scheduler,
    throughput, prefix cache, and state.
19. Mock-server transport/infra: gRPC, Riva, config, app, listener, balancer,
    TLS, main, and lib.
20. Remaining mock-server: metrics, Prometheus, DCGM, accuracy, tests,
    examples, and tools.
21. E2E harness/cellular: common helpers plus cellular, graph, fold, DAG,
    cancellation, warmup, and orchestration tests.
22. E2E endpoints: endpoint-family, gRPC, Riva, KServe, media, multimodal,
    chat, tool-call, completion, ranking, embeddings, and image tests.
23. Remaining e2e: infrastructure, export, telemetry, accuracy, search,
    sweeps, logging, TLS/UDS, stress, and all unassigned e2e tests.
24. Small crates/tools: `rust/loadgen-core/**`, `rust/pyext/**`, and all
    remaining workspace Rust build or tool sources.

Ordering is authoritative: a focused test assigned by an earlier rule does not
fall through to a later catch-all rule.

- [ ] **Step 3: Mark exclusions explicitly**

Move data-like generated or vendor-shaped Rust files to `excluded.txt`, with a
reason per path. At minimum, inspect protocol mirrors, generated constants,
golden modules, and embedded coding corpora before excluding them. SPDX headers
still receive a presence check.

- [ ] **Step 4: Validate one-owner coverage**

The controller must verify:

```text
all_inventory_paths
  == union(agent_01 ... agent_24, excluded)
intersection(any_two_agent_lists) == empty
duplicates_within_each_list == empty
```

Expected: zero missing paths, zero duplicates, and zero overlaps.

- [ ] **Step 5: Store counts**

Record the repository path, baseline path, total file count, owned file count,
excluded file count, and all 24 agent file-list paths and counts in
`manifest.json`. Use the actual values calculated in Steps 1–4; zero or
sentinel values are invalid. Reject the manifest if any owned agent has an
empty list without an explicit controller explanation.

### Task 3: Dispatch the 24 Disjoint Rewrite Agents

**Files:**
- Read: approved design, global constraints, and each agent's exact file list.
- Modify: only files listed in that agent's ownership file.
- Create outside the repository:
  `$baseline/controller/agent-01-report.md` through
  `$baseline/controller/agent-24-report.md`

**Interfaces:**
- Produces: 24 non-overlapping edit sets and reports.
- Consumes: Task 2 manifest and the Graham review skill.

- [ ] **Step 1: Construct one self-contained prompt template**

Each prompt must include:

```text
Read the attached Graham review skill and the approved cleanup design first.
Read every file listed in your ownership file completely.
Edit only those files; never modify another agent's files.
Preserve SPDX, CLI command/flag names, YAML identifiers, contract-critical
citations, safety rationale, and current non-obvious invariants.
Delete code narration and all implementation-history provenance.
Remove `legacy` from non-preserved identifiers and strings, updating every
reference you own.
Do not run workspace-wide formatters.
Write the required report and return only status, changed-file count,
cross-partition references, checks, and concerns.
```

- [ ] **Step 2: Dispatch all 24 agents in one parallel tool call**

Use one subagent per ownership file. Agents operate in the current worktree,
not isolated worktrees, because the required baseline includes uncommitted
changes. The exact one-owner manifest is the write-conflict boundary.

- [ ] **Step 3: Require complete reports**

Each report must list:

- Every file inspected.
- Files changed.
- Files intentionally unchanged and why.
- Excluded/generated-looking files encountered.
- Comments deleted, tightened, and preserved by category.
- Identifiers and strings renamed.
- External wire or artifact changes.
- Cross-partition references needing reconciliation.
- Checks run and results.
- Concerns or blockers.

- [ ] **Step 4: Reject incomplete agents**

An agent is not complete if its report omits an assigned file, edits an
unassigned file, leaves an unexplained historical comment, or reports an
unresolved local reference. Re-dispatch that agent against the same ownership
set after resolving its blocker.

### Task 4: Review Every Partition

**Files:**
- Read: each ownership file, report, and partition diff.
- Modify: partition-owned files only through a focused fix agent.
- Update outside the repository:
  `$baseline/controller/progress.md`

**Interfaces:**
- Produces: 24 approved partitions with no open Critical or Important finding.
- Consumes: Task 3 reports and baseline.

- [ ] **Step 1: Generate partition review packages**

For each agent, create a package containing its exact file list, pre-task
hashes, current diff for those files, and report.

- [ ] **Step 2: Run one focused reviewer per partition**

Review for:

- Compliance with preservation and removal rules.
- Missed files or comments.
- Lost safety, wire, serialization, timing, concurrency, cancellation, or
  allocation rationale.
- Incomplete renames.
- Changed CLI commands, flags, or YAML identifiers.
- Fixture, generated, or embedded-corpus churn.
- Unrelated code changes.

- [ ] **Step 3: Fix and re-review findings**

Dispatch one fix agent per affected partition with the complete finding list.
Re-review until both specification compliance and code quality pass.

- [ ] **Step 4: Record durable completion**

Append one line per clean partition to `$baseline/controller/progress.md`,
including owned-file count, changed-file count, and review status.

### Task 5: Reconcile Cross-Partition Renames

**Files:**
- Read: all 24 reports and the complete Rust tree.
- Modify: only files identified by stale-reference evidence.
- Create outside the repository:
  `$baseline/controller/reconciliation.md`

**Interfaces:**
- Produces: one internally consistent combined tree.
- Consumes: cross-partition reference lists from Task 3.

- [ ] **Step 1: Consolidate rename maps**

Create one old-name to new-name map from all reports. Reject conflicting target
names before editing.

- [ ] **Step 2: Search all code for stale references**

Search every source and test file for each old identifier or string. Classify
matches as preserved CLI/YAML names, excluded data, or stale references.

- [ ] **Step 3: Apply focused reconciliation**

Use one integration agent to update stale references and record every touched
path and reason. Do not perform opportunistic cleanup.

- [ ] **Step 4: Compile-check renamed interfaces**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cargo check -p loadgen-core --all-targets
cargo check -p aiperf-runtime --all-targets
cargo check -p aiperf-cli --all-targets
cargo check -p aiperf-mock-server --all-targets
cargo check -p aiperf-e2e-tests --all-targets
```

Expected: all available default-feature checks pass. Record dependency or
environment blockers verbatim rather than weakening the checks.

### Task 6: Run Repository-Wide Semantic and Lexical Audits

**Files:**
- Read: every Rust source and changed non-Rust reference.
- Create outside the repository:
  `$baseline/controller/prohibited-candidates.txt`
  `$baseline/controller/exceptions.md`

**Interfaces:**
- Produces: reviewed zero-unexplained-match audit.
- Consumes: approved design and combined tree.

- [ ] **Step 1: Scan historical vocabulary**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
baseline="$(< /tmp/aiperf-comment-cleanup-current)"
rg -n -i \
  '\b(legacy|provenance|ported?|porting|migrat(e|ed|ing|ion)|formerly|previously|superseded|absorbed|refactor(ed|ing)?|stage|track|slice)\b' \
  --glob '*.rs' rust \
  > "$baseline/controller/prohibited-candidates.txt"
```

Expected: every match is manually classified. Network ports, portable code,
mathematical terms, preserved CLI/YAML names, SPDX, and contract-critical
citations may remain only with a recorded reason.

- [ ] **Step 2: Scan narrative comment forms**

Search Rust comments for step banners, proof banners, history phrases, stale
crate paths, source line citations, “mirrors X” narration, and comments that
begin with obvious code-echo verbs such as “create”, “set”, “return”, “build”,
or “parse”. Review each candidate semantically; do not delete useful current
constraints solely because they match a heuristic.

- [ ] **Step 3: Verify SPDX coverage**

Check every non-excluded Rust file starts with exactly the required two SPDX
lines. Excluded Rust files must retain their existing required header.

- [ ] **Step 4: Verify preserved interfaces**

Compare CLI command names, flag names, and YAML identifiers with the Task 1
baseline. Any difference is a defect unless it is pre-existing and byte-equal
to the baseline patch.

- [ ] **Step 5: Approve exceptions**

For each remaining candidate, record path, line, category, and reason in
`exceptions.md`. Completion requires zero unexplained candidates.

### Task 7: Format and Verify the Combined Tree

**Files:**
- Read: all changed files.
- Modify: only cleanup-caused formatting defects in cleanup-touched files.

**Interfaces:**
- Produces: buildable, tested, rustdoc-valid combined tree.
- Consumes: Tasks 4–6 approved output.

- [ ] **Step 1: Check formatting without broad writes**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cargo fmt --all -- --check
```

Expected: pass. If it fails, compare against the Task 1 baseline and format
only cleanup-touched files responsible for new failures.

- [ ] **Step 2: Run focused unit tests**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cargo test -p loadgen-core
cargo test -p aiperf-runtime --lib
cargo test -p aiperf-cli --lib
cargo test -p aiperf-mock-server
```

Expected: all tests pass.

- [ ] **Step 3: Compile all integration tests**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cargo test --workspace --no-run
```

Expected: all default-feature workspace tests compile.

- [ ] **Step 4: Check rustdoc links**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

Expected: pass with no broken intra-doc links or rustdoc warnings introduced by
the cleanup.

- [ ] **Step 5: Run feature-bearing checks**

Run available checks:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cargo test -p aiperf-runtime --features dynosim --lib
cargo build -p aiperf-cli --features dynosim
cargo build -p aiperf-cli --features full
```

Expected: pass when the sibling Dynamo checkout and feature dependencies are
available. Report unavailable dependencies as environment limitations.

### Task 8: Perform Final Whole-Tree Review

**Files:**
- Read: approved design, baseline, manifest, reports, exception list, combined
  diff, and verification output.
- Modify: only files required to resolve final findings.
- Create outside the repository:
  `$baseline/controller/final-review.md`

**Interfaces:**
- Produces: final Graham-style approval or actionable findings.
- Consumes: all prior tasks.

- [ ] **Step 1: Dispatch a broad final reviewer**

The reviewer must inspect the complete cleanup delta against the Task 1
baseline, not merely `git diff HEAD`, because the branch contained pre-existing
changes.

- [ ] **Step 2: Validate exhaustive coverage**

Confirm every manifest path is marked reviewed or excluded, every agent report
is complete, and every exception has an approved reason.

- [ ] **Step 3: Validate minimal diff surface**

Reject unrelated behavior changes, unexplained fixture churn, removed
contracts, and formatting-only changes outside cleanup-touched regions.

- [ ] **Step 4: Resolve all findings in one fix wave**

Dispatch one integration fix agent with the complete final finding list, rerun
covering checks, and re-review.

- [ ] **Step 5: Prepare the final user report**

Report:

- Total files reviewed, changed, unchanged, and excluded.
- Comment lines deleted and rewritten.
- Internal identifiers and strings renamed.
- Preserved CLI/YAML names and approved contract citations.
- External wire or artifact changes.
- Verification commands and exact outcomes.
- Remaining environment limitations.

Do not stage or commit changes.
