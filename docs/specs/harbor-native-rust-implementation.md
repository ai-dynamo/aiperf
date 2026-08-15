<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor native-Rust implementation

## Purpose

This record defines the implementation program for the Harbor replacement platform. It implements the replacement bar in `harbor-replacement-platform.md` through a native-Rust evaluation domain and native execution paths. Harbor packages and registry references are import formats only. No Harbor Python runtime, library, wrapper, bridge, or dependency participates in execution.

## Built P0 native composition

The runtime currently provides native local-file and pinned-Git package acquisition through `NativeSourceAcquirer`, with 40-hex-commit and repository-relative-package-path validation. `HarborEvaluationCoordinator` performs source import before sandbox preflight/open, then prepares the declared verifier artifact transfer. Unsupported package semantics therefore return from import before environment opening.

P0 acceptance tests cover a local source lifecycle with an external contract, deterministic pinned-Git source identity after repository `HEAD` changes, and a separately provisioned verifier receiving only declared artifacts at their exact paths. The test suite links only native Rust crates; no Harbor executable, runtime, library, wrapper, bridge, or dependency is present in the P0 path.

## Architecture boundary

The runtime owns a new `eval` domain. Its types are distinct from cellular transport DTOs and from the narrow replay `GraphTraceProgram` execution representation.

```text
Harbor-compatible source
  -> acquisition + immutable source artifact
  -> importer + ImportReport
  -> eval::{TaskSpec, EvalDatasetManifest, TrialSpec}
  -> sandbox + AgentContract + verifier
  -> append-only Attempt evidence
  -> ScoreVersion / regrade / paired comparison
```

`eval` owns immutable identities, digests, provenance, source artifacts, import/lowering reports, trials, attempts, evidence spans, scores, and regrade requests. It owns no process, provider, or sandbox implementation.

`GraphTraceProgram` remains a measured execution program. It is not the semantic source of truth for Harbor tasks or native agent workflows. Semantic graphs lower fallibly into executable programs and carry a fidelity report.

## Import and trial contract

Acquisition supports local directories, pinned Git revisions, and Harbor-style registry references. The importer preserves the source package byte-for-byte, then produces a normalized task/dataset projection plus a machine-readable report with one of `native`, `lossless_normalized`, `lossy_normalized`, or `unsupported`.

Unsupported semantics stop the trial before environment provisioning. Importing never weakens verifier isolation, egress policy, artifact transfer restrictions, or multi-step continuation requirements.

A resolved trial pins task digest, agent/graph variant, model parameters, seed, policy, resource budget, environment and verifier identities, source revision, and runtime configuration. Equal resolved inputs reproduce trial and artifact-manifest identity. A rerun creates a new attempt; it never overwrites evidence or a prior score.

## Execution, policy, and verification

An `AgentContract` is `External`, `Installed`, or `NativeGraph`. Capability preflight resolves the selected sandbox/provider/image/recipe before environment spend and fails closed if persistent workspaces, read-only base/overlay isolation, staging, network controls, secret policy, descendant termination, or resource guarantees cannot be met.

Native graph branches run in copy-on-write overlays or clones. They return immutable patches/artifacts. Only an explicit selector or merge operation can advance the canonical workspace.

Verifiers run either in the task sandbox when declared shared, or in a fresh sandbox/restored snapshot when separately provisioned. Separate verifiers receive declared artifact copies at declared paths and permitted evidence only; they never receive ambient agent credentials, mutable agent workspace state, or the agent control channel.

Reward parsing prefers `reward.json` to `reward.txt`, supports multi-metric finite numeric rewards, and records malformed or absent values as invalid verifier evidence. Evidence distinguishes task validity, agent outcome, replay fidelity, and system performance. No aggregate score conflates these categories.

A regrade pins a verifier identity and preserved attempt evidence. It appends a new `ScoreVersion` with its rationale and never changes the original score.

## Parallel delivery tracks

1. **Recorded replay closure:** complete Task 12 cellular preflight/supplement/artifact shipping and Task 13 product/A-B parity. Its controller-only artifacts remain reusable infrastructure, not Harbor evidence authority.
2. **Evaluation identity and reports:** establish the isolated `eval` domain, strict DTOs, source digests, provenance, import reports, trial/attempt identity, evidence records, and score versions.
3. **Acquisition and importer:** implement local, pinned-Git, and registry-reference import; source preservation; strict normalization; and unsupported-semantic refusal.
4. **Sandbox and agent contracts:** implement Harbor recipes, external/installed contracts, overlay workspaces, capability preflight, resource/network/secret policy, and provider fakes.
5. **Verifier, rewards, and regrade:** implement shared/separate verifier execution, artifact-only handoff, reward parsing, immutable evidence, versioned scores, and regrade.
6. **Semantic lowering and experiments:** add semantic graph/lowering/fidelity validation and paired quality/cost/latency/critical-path comparison against fixed trial baselines.
7. **P1/P2 evolution:** add multi-step continuation, provider negotiation, task health/quarantine, registry/share workflows, and training-data export. These tracks may develop in parallel but integrate only through the immutable `eval` contracts.

Track 2 is the stable contract boundary for Tracks 3–6. P0 compatibility acceptance is the first integration gate: native local and pinned-Git import, external or installed agents, shared and separate verifier modes, declared-artifact isolation, deterministic identity, and regrade. P1/P2 cannot weaken those P0 rules.

## Verification requirements

Every track supplies strict unit tests and deterministic fake-provider/sandbox tests. P0 product fixtures prove native local and pinned-Git source acquisition, an external agent contract, separately provisioned verifier declared-artifact isolation, exact declared artifact paths, deterministic source identity, reward precedence, and immutable regrade. Unsupported import refusal before sandbox opening is covered by the coordinator contract suite.

Semantic experiments prove a native graph variant report with paired quality, cost, latency, and critical-path deltas while task, model, seed, policy, image, and budget remain fixed.

P1/P2 verification covers multi-step and provider capability behavior, task health and quarantine, registry/share semantics, and trajectory export without requiring online registry availability for local/private suites.

## Non-negotiable constraints

- All Harbor implementation is native Rust.
- The `eval` namespace prevents collision with existing cellular `DatasetManifest` transport types.
- Controller-only replay artifact folding does not replace append-only Harbor attempt evidence.
- Separate-verifier isolation does not reuse a shared replay sandbox by default.
- Unsupported import, lowering, transform, and provider capability paths are typed refusals, never silent fallback.
- No Harbor completion claim is valid until P0 acceptance tests run without a Harbor runtime.
