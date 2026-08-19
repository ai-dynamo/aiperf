<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor externally driven compatibility runner

## Purpose

Enable one `externally_driven` NativeGraph task to complete a Harbor episode
without presenting an opaque user-language agent as a Rust-controlled
NativeGraph policy. Rust remains authoritative for immutable task selection,
environment lifetime, artifact collection, verifier execution, reward, score,
and outer timing. The external driver remains an untrusted supervised process.

The result must be visibly lower fidelity than `native_graph`: it is always
`externally_driven` and is never eligible to claim native scheduling, model
control, full tool visibility, or exact NativeGraph fidelity.

## Decision

Three approaches were considered.

1. Run the configured driver through the existing generic agent-command path.
   This is rejected because it supplies model-secret machinery and has no
   correlated terminal-candidate protocol or sealed compatibility evidence.
2. Treat the external process as a NativeGraph policy adapter. This is
   rejected because it would incorrectly imply Rust controls the driver’s
   model decisions and could grant native-profile authority.
3. Add a narrow compatibility runner around the existing external-driver
   registry. This is selected. It uses a separate external-only authorization,
   a Driver-role protocol session, terminal-only evidence, and the existing
   Harbor verifier and episode coordinator.

## Immutable preflight

An externally driven package must retain an exact
`external_driver_factory_id`, the declared driver adapter, its argv, and all
package identity material. Before Docker provisioning, the CLI resolves that
exact factory from the frozen application registry and rejects unknown,
normalized, or mismatched registrations.

The factory evolves from `bind(package)` into two stages:

1. `prepare(package, resolved_trial)` validates immutable package and trial
   facts without access to a process, environment, or spawn capability.
2. `run(session)` executes only after the runner has acquired the task
   environment and minted a constrained compatibility session.

The prepared driver cannot choose a different package, task, environment,
trial, attempt, driver argv, or deadline. `--agent-command` remains refused.
Single-task external execution does not accept or require `--model-runtime`.
Suite mode remains fail-closed until its model-binding grammar has an explicit
external-compatible contract.

## Supervised driver session

The runner mints `ExternallyDrivenAdapterAuthorization` only from the imported
external package, resolved trial, and declared driver adapter. It can create
one Driver-role adapter request with the declared argv, an empty sanitized
environment, a bounded driver deadline, and no model-secret mapping. It does
not grant NativeGraph’s exact-profile authorization or `NoAdapterEgress`
claim.

Docker exposes a distinct compatibility spawner and an isolated
`execute_externally_driven_with_runtime` transaction. It refuses Compose,
multi-step recipes, missing driver selection, missing prepared driver, and
incompatible protocol/runtime configuration before build or environment spend.
The successful transaction is:

```text
preflight -> acquire environment -> healthcheck -> start Driver session
          -> correlated terminal receipt -> declared artifacts -> verifier
          -> score -> reverse cleanup
```

Any driver error, timeout, cancellation, missing terminal receipt, invalid
receipt, or artifact/verifier failure cancels and reaps the driver exactly once.
No terminal receipt means the verifier is not invoked.

## Terminal protocol and evidence

The compatibility session builds one Driver-role `AdapterProtocolConfig` with
only the Driver capability. It sends `RequestEpisodeTerminal` and accepts one
correlated terminal candidate from that driver session. The candidate is
bounded and converted immediately to an opaque canonical digest. Raw protocol
JSON, stdout, stderr, traffic, paths, handles, prompts, tool data, and secrets
do not enter public APIs, artifacts, verifier input, errors, metrics, or
frozen evidence.

`CapturePolicy` remains package-bound and `CompatibilityObservation` remains
digest-only with its fixed 1,024-call cap. A zero-observation run freezes as
`Missing`, never `ObservedProxy`. The compatibility completion seals the
package/trial/attempt/capture-policy identity and appends exactly one
`EvidenceKind::Compatibility` lifecycle event. It does not modify verifier
evidence, reward, score, or score lineage.

## Result classification

`EpisodeResult` receives an explicit fidelity axis. NativeGraph results retain
their existing native classification. Compatibility results carry
`ExternallyDriven(ObservedProxy | Partial | Missing)`. Legacy construction is
non-exact and cannot silently become native-controlled.

The completed-attempt boundary accepts a compatibility supplement only when
the resolved trial selected the external profile and all package, source, task,
environment, trial, attempt, and capture-policy facts match. A native or
no-profile attempt rejects the supplement; an external attempt rejects a
native-rollout supplement.

## Outer composition

`DockerExternallyDrivenEpisodeExecutor` implements the existing
`NativeGraphEpisodeExecutor`. It reuses the existing NativeGraph episode
runner, selected evaluator, Harbor evaluation coordinator, scheduler, matrix,
artifact collection, verifier, reward, and score aggregation. It does not add
a second scheduler, graph executor, result authority, or transport client.

The CLI’s external single-task branch changes only after preflight succeeds:
it constructs the prepared compatibility execution and runs one resolved trial
with no Rust-owned model units. The current generic compatibility-runner
refusal remains the behavior for unsupported suite input and any unimplemented
or incompatible selected component.

## Delivery sequence and acceptance tests

### A. Pure contracts

- Require and identity-bind the external factory selector and declared driver.
- Verify exact registry matching and fail before provisioning for unknown or
  normalized selectors.
- Make zero capture classify as `Missing`.
- Seal the terminal supplement and its compatibility-only lifecycle event.
- Add the result fidelity axis while preserving verifier evidence, reward, and
  score lineage.

### B. Supervised backend

- Prove preparation precedes build and receives no spawn capability.
- Prove the spawned driver has only declared argv, empty environment, and no
  model secret access.
- Prove correlated terminal receipt, artifact collection, verifier, and
  cleanup ordering.
- Prove malformed, absent, or timed-out terminal receipt skips verifier and
  reaps exactly once.
- Prove Compose, multi-step, and missing-spawner plans refuse before build.

### C. Runner and CLI

- Prove one external task completes through the actual Harbor coordinator and
  reports externally driven fidelity.
- Prove unknown factory, mismatched argv, and `--agent-command` fail before
  Docker provisioning.
- Prove no model runtime is required for the external single-task path.
- Prove suite mode stays explicitly refused.

## Deliberate exclusions

This work does not add a capture proxy, direct evaluator client, model resolver
or model secrets for the driver, GraphTrace/live-driver control, dynamic
controls, RL/cellular transport, Compose support, remote source transfer, or a
generic Docker rewrite. Optional traffic capture remains a later observation
feature and cannot become an execution or score authority.
