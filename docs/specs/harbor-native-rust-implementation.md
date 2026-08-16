<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor native-Rust implementation

## Purpose

This record defines the implementation program for the Harbor replacement platform. It implements the replacement bar in `harbor-replacement-platform.md` through a native-Rust evaluation domain and native execution paths. Harbor packages and registry references are import formats only. No Harbor Python runtime, library, wrapper, bridge, or dependency participates in execution.

## Built P0 native composition

The runtime currently provides native local-file and pinned-Git package acquisition through `NativeSourceAcquirer`, with 40-hex-commit and repository-relative-package-path validation. Import acquires once into an owned canonical source snapshot; normalization and execution never reread the caller's path. Directory snapshots retain sorted regular-file and directory entries, canonical `0644`/`0755` modes, bytes, and empty directories while rejecting links and special or non-UTF-8 entries. The source digest covers that complete artifact for provenance. The current v2 package identity combines one canonical normalized plan with its exact executable-source projection: standard tasks select the complete `environment/` build context and every resolved verifier test tree, directory-backed JSON packages select their complete tree, and file JSON packages select the primary file. Plan identity includes commands, all resolved execution policy, and sorted unique artifact exclusions. `EvalTaskRef.digest`, `ImportReport.normalized_digest`, and `HarborTaskPackage::identity_digest()` are the same package digest.

A standard manifest may configure paired `[agent].timeout_sec` and `[verifier].timeout_sec` values; both must be finite and positive. The default task-directory executor builds the environment in Docker with no runtime network, runs the external agent before verifier files are copied into the container, then runs shared verification and reads the reward from `/logs/verifier`. Docker and local execution use only private materializations of the retained snapshot, so caller mutation or removal after import cannot affect a build, JSON fixture, or selected verifier tree. Docker applies the configured agent and verifier limits independently to their respective phases and force-removes the timed-out phase container before it returns the timeout result. The explicit `--sandbox local` temporary-root backend remains a deterministic test backend, does not enforce manifest timeouts, and refuses explicit multi-step layouts because it cannot provide their Docker lifecycle guarantees. `HarborEvaluationCoordinator` performs source import before sandbox preflight/open, then prepares the declared verifier artifact transfer. Unsupported package semantics therefore return from import before environment opening.

P0 acceptance tests cover a local source lifecycle with an external contract, deterministic pinned-Git source identity after repository `HEAD` changes, a Docker standard-directory lifecycle where the agent cannot see verifier files, and a separately provisioned verifier receiving only declared artifacts at their exact paths. The test suite links only native Rust crates; no Harbor executable, runtime, library, wrapper, bridge, or dependency is present in the P0 path.

## Built benchmark multi-step subset

Schema `1.0` standard-task manifests may use ordered `[[steps]]` entries. Each step names `steps/<name>/instruction.md` and selects `steps/<name>/tests/test.sh` when present, otherwise the root `tests/test.sh`. Root agent, verifier, and artifact policy is fully resolved into every immutable step plan before provisioning; supported step fields overlay phase environment, user, network, and timeout policy plus verifier mode/environment and step artifacts. Invalid names, effective artifact collisions, unsupported fields, invalid timeouts, and provider capabilities fail before Docker effects.

Docker builds the task image once, starts one persistent agent container, and reuses the external agent command with only the current step instruction injected. After each successful agent phase it copies that step's declared artifacts into a distinct host snapshot before installing a fresh selected test tree. A shared verifier uses the agent container; a separate verifier receives only the immutable artifact snapshot in a new container and never mounts the mutable agent workspace. Test and reward state is cleared at verifier boundaries. The first agent, collection, or verifier error stops every successor step, while cleanup still attempts every acquired container and its anonymous volumes.

The executor reserves `/tests` and `/logs/verifier` for shared-verifier state. If any resolved step uses a shared verifier, an authored, CLI, or inspected image workdir equal to or below either namespace fails closed; directional ancestors such as `/` and `/logs` remain valid. Authored workdirs fail during import, CLI workdirs fail before build, and an implicit image workdir is inspected immediately after the persistent agent container starts and before healthcheck or phase work. Separate-only plans retain their independent artifact-staging collision validation.

`multi_step_reward_strategy = "mean"` averages the union of finite reward metrics across successful steps, treating a missing metric as zero; `"final"` returns the final step reward unchanged. CLI JSON adds ordered `steps` with each step's name, artifacts, and reward. Its top-level reward is the selected aggregate and its top-level artifacts are the final step's artifacts. Implicit single-step and legacy JSON output retain the existing three-key contract without `steps`.

This subset is benchmark execution only. A strict Docker Compose sidecar environment is available to standard tasks: the required Dockerfile remains the runtime-owned `main` image source, while the exact `environment/docker-compose.yaml` overlay can define only validated sidecars and `main.depends_on`. Public-network Compose projects preflight through read-only canonical configuration before provider mutation, use a task-owned lease, and preserve the Dockerfile authority over `main`. Service evidence and argv collection hooks are separate-verifier-only and final-step-only; main stops before non-main evidence collection, and only the frozen declared artifacts cross the verifier boundary. See [benchmark-compose-environments.md](benchmark-compose-environments.md) for the enforced subset.

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

Acquisition supports local directories and pinned Git revisions; the `eval` CLI materializes a pinned local or remote repository into an owned checkout before import. Harbor-style registry references are parsed as an import format but native acquisition deliberately returns `Unavailable` for them. The importer preserves file packages byte-for-byte, captures local directories into one owned tree, and produces a normalized task/dataset projection plus a machine-readable report with one of `native`, `lossless_normalized`, `lossy_normalized`, or `unsupported`. Full-source provenance and normalized package identity are intentionally distinct: comments and unused standard-task files change the source digest but not the package digest when the normalized plan and executable projection are unchanged.

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
3. **Acquisition and importer:** implement local and pinned-Git import; preserve source; strictly normalize; and refuse unavailable registry acquisition and unsupported semantics.
4. **Sandbox and agent contracts:** implement Harbor recipes, external/installed contracts, overlay workspaces, capability preflight, resource/network/secret policy, and provider fakes.
5. **Verifier, rewards, and regrade:** implement shared/separate verifier execution, artifact-only handoff, reward parsing, immutable evidence, versioned scores, and regrade.
6. **Semantic lowering and experiments:** add semantic graph/lowering/fidelity validation and paired quality/cost/latency/critical-path comparison against fixed trial baselines.
7. **P1/P2 evolution:** extend the built benchmark multi-step subset with provider negotiation, richer service orchestration, task health/quarantine, registry/share workflows, and training-data export. These tracks may develop in parallel but integrate only through the immutable `eval` contracts.

Track 2 is the stable contract boundary for Tracks 3–6. P0 compatibility acceptance is the first integration gate: native local and pinned-Git import, external or installed agents, shared and separate verifier modes, declared-artifact isolation, deterministic identity, and regrade. P1/P2 cannot weaken those P0 rules.

## Verification requirements

Every track supplies strict unit tests and deterministic fake-provider/sandbox tests. P0 product fixtures prove native local and pinned-Git source acquisition, an external agent contract, separately provisioned verifier declared-artifact isolation, exact declared artifact paths, deterministic source identity, reward precedence, immutable regrade, timeout-pair normalization, and Docker timeout cleanup for agent and separate-verifier containers. Unsupported import refusal before sandbox opening is covered by the coordinator contract suite.

Semantic experiments prove a native graph variant report with paired quality, cost, latency, and critical-path deltas while task, model, seed, policy, image, and budget remain fixed.

The built multi-step verification covers immutable planning, persistent Docker workspace state, per-step test and verifier isolation, immutable artifact snapshots, exact source-context identity and materialization after origin removal, canonical modes and empty directories, directional shared-workdir refusal, phase-policy inheritance/overrides, mean/final rewards, additive CLI output, terminal failure, eager cleanup, and strict Compose preflight, service-evidence ordering, and labelled cleanup. Future P1/P2 verification covers broader provider capability behavior, task health and quarantine, registry/share semantics, and trajectory export without requiring online registry availability for local/private suites.

## Non-negotiable constraints

- All Harbor implementation is native Rust.
- The `eval` namespace prevents collision with existing cellular `DatasetManifest` transport types.
- Controller-only replay artifact folding does not replace append-only Harbor attempt evidence.
- Separate-verifier isolation does not reuse a shared replay sandbox by default.
- Unsupported import, lowering, transform, and provider capability paths are typed refusals, never silent fallback.
- No Harbor completion claim is valid until P0 acceptance tests run without a Harbor runtime.
