<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# Recorded-agent replay and Harbor handoff

## Branches

- Active worktree branch: `ajc/rust-open-lab` at `b692ee59a6`.
- Harbor follow-on branch: `ajc/rust-harbor` at `3d3146df00`, created from the
  OpenLab line without switching this worktree.

The Harbor branch should use these repository records as its source of truth:

- `docs/specs/harbor-replacement-platform.md` — pure native-Rust Harbor
  replacement bar and P0 acceptance criteria.
- `docs/specs/agentic-eval-platform.md` — native task, trial, evidence, and
  verifier architecture.
- `docs/specs/semantic-agent-graph.md` — graph-agent and fidelity boundary.

Do not introduce a Harbor Python runtime, bridge, wrapper, or dependency. The
P0 path is a native-Rust importer, normalized task/trial model, sandbox/agent/
verifier execution, immutable evidence, scoring, and regrading.

## Recorded-agent replay ledger

The active ledger is
`.superpowers/sdd/2026-08-14-recorded-agent-replay/progress.md`; the full plan
is `/home/anthony/.aiperf/docs/superpowers/plans/2026-08-14-recorded-agent-replay.md`.

Tasks 1–10 are completed and independently reviewed. Task 11 is implemented
through `b692ee59a6` and its final follow-up review is approved:

- controller-owned atomic checkpoints and redacted provenance;
- seedless resume restores its protected namespace before validation;
- request-plan and resolved-environment identity changes invalidate resume;
- terminal trace completion writes durable checkpoint progress during phase
  execution, before a later trace can fail the phase;
- legacy checkpoint completion-map data migrates to the version-2 vector form;
- duplicate task identities and unsupported checkpoint versions fail closed;
- `resume && cells > 1` is rejected, while the identity cell partition `(0,1)`
  is allowed;
- canonical `recorded-agent-default` validation compares the actual loaded
  bundle, not only fixture defaults.

Tasks 12 and 13 remain pending. Do not mark the ledger complete until they are
implemented, independently reviewed, and the final product/documentation
verification is run.

## Recent verification

Task 11 final focused evidence included:

- `cargo test -p aiperf-runtime --test recorded_agent_resume` — 6 passed.
- Engine entrypoint partition regression — passed.
- Engine checkpoint callback regression — passed.
- `cargo check -p aiperf-runtime --features engine` — passed.

The repository's advertised `.venv/bin/activate` is absent in this worktree;
focused Rust checks used `RUSTC_WRAPPER=` when sandboxed `sccache` was
unavailable. Existing dead-code warnings are unrelated.

## Worktree hygiene

Preserve these pre-existing unrelated changes while continuing:

- modified `docs/specs/README.md`, `docs/specs/agentic-eval-platform.md`, and
  `llms.txt`;
- untracked `docs/specs/dag-v3-graph-ir-extraction.md`,
  `docs/specs/harbor-replacement-platform.md`, and
  `docs/specs/semantic-agent-graph.md`;
- untracked `rust/cli/core.8130`.
