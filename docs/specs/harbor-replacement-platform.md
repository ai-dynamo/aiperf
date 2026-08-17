<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor replacement platform

## Purpose

This record defines the replacement bar for Harbor-style agent evaluation.
AIPerf shall execute imported Harbor-compatible task suites without Harbor's
runtime and add native graph experimentation, production-workflow replay, and
explainable performance evidence.

All implementation is pure native Rust. Harbor task directories, manifests,
and registry references are import formats only. No Harbor Python process,
library, wrapper, or permanent bridge participates in execution.

## Replacement bar

AIPerf replaces Harbor for the built compatibility tier when a user can run an
unchanged supported Harbor task from a local path or pinned Git reference;
choose an external or installed agent; receive reproducible scores and
artifacts; regrade the attempt; and never require Harbor's runtime. Native
graph-agent execution beyond the bounded acyclic NativeGraph slice and online
registry acquisition are future extensions. The built slice requires a strict
host model-runtime mapping, a native-graph lifecycle request, a standard
adapter-free task, and `no-network` task/agent Docker phases; it uses
host-owned AIPerf model transport and a plan-bound Docker `NoAdapterEgress`
proof for one scored matrix trial.

```text
Harbor-compatible package
  -> Rust importer + preserved source artifact
  -> normalized TaskSpec / DatasetManifest
  -> native TrialSpec matrix
  -> AIPerf sandbox + agent + verifier lifecycle
  -> immutable evidence, scores, comparison, and regrade
```

## Built

The agentic evaluation platform design already defines immutable task, dataset,
trial, sandbox, verifier, evidence, score, task-health, and paired graph
experiment concepts. The native runtime supplies graph execution, clocked
placement, cellular result folding, and controller-owned artifact seams.

## Observed Harbor use cases and replacement priority

Harbor does not publish a usage-share breakdown. The following ordering is the
visible/documented use-case hierarchy and therefore the compatibility priority,
not a claim about measured customer usage:

1. **Evaluate coding and terminal agents on established benchmarks.** Users run
   common agents against Terminal-Bench, SWE-Bench, Aider Polyglot, and similar
   containerized suites.
2. **Run private or local task suites.** A local directory or pinned Git
   dataset is evaluated without first publishing it.
3. **Compare agent, model, prompt, and configuration variants.** One job
   expands a trial matrix and retains results for comparison.
4. **Scale sandboxed rollouts.** Long-lived, I/O-bound trials run concurrently
   through local or cloud environment providers.
5. **Create, adapt, version, and publish benchmark datasets.** Adapters and
   oracle runs establish task correctness before a suite is shared.
6. **Produce post-training data.** Agent trajectories are exported as rollouts
   or success/failure training splits.
7. **Share and review jobs.** Teams inspect trajectories, verifier output,
   artifacts, and comparisons, then resume or distribute partial results.

P0 therefore covers items 1–3 for local/pinned sources. P1 adds scalable
provider execution and benchmark-adaptation workflows. P2 adds publishing,
sharing, and training-data export. The native graph/evidence model improves all
three phases; it must not delay the simple P0 task-to-verifier loop. Harbor P0
is an early vertical proof of the replacement claim, not a dependency deferred
until a universal semantic agent ontology exists.

## Compatibility tiers

### P0: normal single-step evaluation

The importer shall acquire one owned canonical source snapshot and normalize
from that snapshot only. Standalone package files are preserved byte-for-byte;
directory snapshots preserve canonical entry paths, kinds, modes, bytes, and
empty directories:

- task instruction, task metadata, dataset manifest, environment build/image,
  test/verifier scripts, optional solution, and task artifacts;
- task and verifier users/timeouts, Linux/Windows target, resources, network
  policy/allowlists, environment variables, MCP declarations, health checks,
  secrets policy, and artifact collection rules;
- local directories and pinned Git revisions. Harbor-style registry references
  are recognized source syntax but currently refuse with a typed unavailable
  result; they are not acquired through Harbor or Python fallback;
- `reward.json` then `reward.txt`, including multi-metric numeric rewards;
- external environment-driving agents and installed/headless agents. The
  `NativeGraph` contract is executable for the bounded acyclic adapter-free
  slice above; dynamic controls, adapters, externally driven profiles, and
  multi-lifecycle suites remain typed refusals.

P0 supports shared and separately provisioned verifier modes. A separate
verifier receives declared artifacts at declared paths, never ambient agent
credentials or workspace state.

### Built ordered benchmark subset

Schema-`1.0` standard tasks may author an ordered `[[steps]]` layout. Native
Docker execution builds once, preserves one agent workspace across steps,
snapshots declared artifacts before every verifier, and gives each verifier a
fresh selected test tree. Shared verifiers use the persistent agent container;
separate verifiers receive only the immutable artifact snapshot in an isolated
container. Root phase policy is inherited unless a supported step field
overrides it. `mean` and `final` reward strategies produce an aggregate reward,
and the CLI adds ordered per-step reward/artifact data without changing the
implicit single-step JSON contract. A terminal step failure prevents successor
work and cleanup attempts every acquired container.

The complete captured artifact is provenance. Normalized package identity is a
versioned digest of the canonical resolved plan and exact executable-source
projection: the full standard-task `environment/` tree plus selected test
trees, the full directory-backed JSON tree, or the standalone JSON file.
Artifact-exclusion lists are sorted and deduplicated because their execution is
an unordered disjunction. Docker and local execution materialize only the
retained snapshot, so mutation or removal of the import origin cannot alter an
execution.

When any resolved step uses a shared verifier, the persistent agent workdir
cannot equal or descend from `/tests` or `/logs/verifier`. Authored manifests,
CLI overrides, and implicit image workdirs all enforce the same directional
rule; ancestors such as `/` and `/logs` remain valid, and separate-only plans
retain separate artifact-staging checks.

This is a benchmark-execution subset, not general service orchestration.
Standard tasks may add the exact `environment/docker-compose.yaml` sidecar
overlay to their required Dockerfile. AIPerf generates and owns `main`,
strictly validates the sidecar-only overlay and its canonical Compose result,
and runs the project only on its public network. Sidecar evidence is separate
verifier-only and final-step-only; the frozen declared artifact transfer is the
only path from a Compose project into its verifier. General Compose
passthrough, host-facing resources, restricted Compose networking, and
arbitrary service orchestration remain unsupported.

### P1: fuller task and provider parity

The built ordered subset expands through setup hooks, richer trial aggregation,
composite manifests, provider capability negotiation, service orchestration,
trajectory-aware verifiers, accelerator declarations, and policy-controlled
judge credentials after P0 evidence is stable.

### P2: distribution ecosystem

Native registry publication, sharing, and cloud-provider expansion follow
local/private execution. Online registry availability is never required for a
local or private suite.

## Native task, agent, and verifier contracts

`TaskSpec` owns instruction, environment, verifier, resource budget, network
and secret policy, artifacts, source provenance, and capability requirements.
The model has an `AgentContract` vocabulary of:

- `External`: AIPerf drives a constrained environment interface;
- `Installed`: a headless agent executes inside the task environment; or
- `NativeGraph`: a planned AIPerf-owned planner/tool/delegation topology.

The built CLI execution path accepts external commands and package-installed
commands. It rejects a lifecycle request selecting `NativeGraph` because no
native graph-agent executor is wired to the evaluation sandbox yet. The
existing semantic lowering and paired comparison contracts are fidelity and
analysis primitives; they do not make a native graph a runnable P0 agent.

Provider capabilities for the built external and installed paths are negotiated
and fail closed before environment spend. A future native graph executor must
use explicit overlay/clone workspaces and return immutable candidate
patches/artifacts; only a selector/merge operation may advance the canonical
workspace.

The verifier uses a fresh sandbox or restored snapshot. Functional tests,
properties, metamorphic checks, negative controls, and security policy checks
produce independent typed findings rather than a gold-patch-only verdict.

## Import, evidence, and regrade

Every imported package has an immutable source artifact, normalized package
digest, and machine-readable report. The source digest covers the complete
captured artifact, while the normalized digest binds the canonical execution
plan and exact executable-source projection:

```text
native | lossless_normalized | lossy_normalized | unsupported
```

Unsupported semantics fail rather than weakening verifier isolation, egress
policy, artifact transfer, or multi-step continuation. The report is part of
trial identity.

A trial records task, agent, model, graph variant, seed, policy, environment,
attempt, verifier output, reward, score rationale, patch/artifact references,
and full event evidence. A subsequent grade runs a pinned verifier against preserved
attempt evidence; it creates a new versioned score rather than overwriting the
trial.

## Native differentiators

Harbor compatibility is the entry point, not the product limit. AIPerf adds:

- graph variant as a first-class trial dimension with paired causal comparisons;
- node-level critical-path, token, cost, latency, cache, and tool-resource
  attribution;
- one evidence model for live task trials and imported production workflow
  replay, with distinct truth claims;
- task-health, reference solve, negative control, drift quarantine, and
  versioned score revision; and
- policy/security spans and branch-isolated workspace evidence.

No aggregate score may conflate task validity, agent quality, replay fidelity,
or system performance.

## Acceptance requirements

1. A representative unchanged P0 Harbor task executes without Harbor runtime
   from both a local package and a pinned Git source.
2. The same task supports at least one external or installed agent contract in
   both shared and separate verifier modes.
3. Separate verifier execution cannot read agent credentials or undeclared
   workspace state.
4. Declared artifacts materialize at exact verifier paths.
5. Reward parsing matches `reward.json`/`reward.txt` semantics.
6. Unsupported import semantics fail with a report before environment spend.
7. The same canonical plan and executable-source projection reproduce package
   identity; full-source provenance remains independently reproducible.
8. A pinned verifier can regrade preserved attempt evidence without overwriting
   the original score.
9. Import followed by caller mutation or removal executes the retained build
   context and verifier trees, including empty directories and executable modes.

Native graph-agent execution, online registry acquisition, and a native graph
variant product report are future acceptance requirements. They are excluded
from the built P0 compatibility gate rather than being represented by a
fallback or a partial claim.

## Source anchors

- `docs/specs/agentic-eval-platform.md` — task, trial, evidence, and health
  architecture.
- `docs/specs/semantic-agent-graph.md` — native graph and fidelity boundary.
- `docs/specs/recorded-agent-replay-rust-port.md` — initial executable path.
- [Harbor core concepts](https://www.harborframework.com/docs/core-concepts) —
  task/dataset/trial/job vocabulary.
- [Harbor agents](https://www.harborframework.com/docs/agents) — external and
  installed agent compatibility targets.
- [Harbor datasets](https://www.harborframework.com/docs/datasets) — local,
  published, Git-pinned, and composite dataset workflows.
- [Harbor evals](https://www.harborframework.com/docs/run-jobs/run-evals) —
  benchmark execution, artifacts, viewer, and comparison workflows.
- [Harbor adapters](https://www.harborframework.com/docs/datasets/adapters) —
  oracle validation and benchmark adaptation workflows.
- [Harbor SFT workflows](https://www.harborframework.com/docs/training-workflows/sft)
  — trajectory and rollout export workflows.
