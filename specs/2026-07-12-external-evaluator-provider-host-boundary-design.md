<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf evaluator providers: a Rust-owned effect boundary

**Date:** 2026-07-12
**Author:** Anthony Casagrande (Tech Lead) + Codex
**Status:** decided / not implemented — adversarially reviewed and default-refute adjudicated
**Decision:** converge static and stateful accuracy on one provider-neutral evaluation workload,
then retire AIPerf's duplicate Python evaluator/provider implementations after benchmark-by-
benchmark parity gates. NeMo Evaluator and OpenBench/Inspect AI are sibling canonical evaluator
providers behind the same contract; neither wraps or owns the other. The selected provider owns
evaluation semantics. AIPerf Rust owns every upstream/external HTTP, network, and SSE operation,
plus admission, routing, retries, cancellation, accounting, and artifact sealing. The integration boundary is a
typed, supervised pipe protocol with an optional scoped Rust-owned man-in-the-middle compatibility
proxy. Evaluators may use local HTTP/SSE only to that per-run proxy; Rust alone owns every
upstream/external connection and real credential. The boundary is never an endpoint lease,
unscoped Internet proxy, monkeypatch, or Python client of a real model endpoint.

**Research baselines:**

- AIPerf commit `fdd079983f0ee546e71db88784dae995604a268f` plus the local working tree on
  2026-07-12. The working tree contains in-flight protocol-v2 agentic execution that is ahead
  of parts of the written architecture record; this RFC treats code as truth and does not claim
  that those uncommitted changes are shipped.
- NVIDIA-NeMo/evaluator commit `a668af906b46c802984f2d471f15ca83b763092d`, package version
  `0.4.0`.
- groq/openbench commit `3f190a835f7fee34ccd96e17242a36a29e0620a6`, package version
  `0.5.3`.
- Inspect AI `0.3.141`, commit `bb78d82dde311b68dbfd0b49f3186b9fc13a1465`, pinned by
  OpenBench.

**Companions:**

- `2026-07-10-aiperf-rust-accuracy-accumulator-design.md` records the current static and
  stateful evaluator implementation. This RFC proposes its target replacement, not a claim
  about today's code.
- `2026-07-11-aiperf-runner-only-execution-surface-design.md` owns runner protocol v2,
  backend/workload registration, product reachability, and subprocess proof.
- `2026-07-11-python-orchestrator-rust-single-run-design.md` owns the Python CLI and fresh
  `aiperf-runner` process boundary.
- `2026-07-11-aiperf-runner-owned-endpoint-registry-design.md` owns endpoint profiles,
  prepared endpoint bindings, and endpoint capability validation.
- `2026-07-10-aiperf-transport-rust-port-design.md` owns DNS/TCP/TLS/HTTP/SSE behavior.
- `2026-07-10-aiperf-rust-dataset-segment-seam-design.md` owns Rust-side immutable asset
  acquisition and materialization.

---

## 0. Executive decision

The end state is **staged full replacement of AIPerf's Python evaluation implementations by a
replaceable evaluator-provider registry**, not an adapter hidden behind today's
`AgenticHarnessProvider`. The stock providers are `nemo_evaluator` and `openbench`; more providers
may implement the same protocol. The replacement deliberately stops at the semantic boundary. It
does not replace AIPerf Rust's transport, scheduling, endpoint preparation, metrics, supervision,
or report authority.

```text
Python aiperf CLI
    |
    | Config v2; no inference or evaluator networking
    v
aiperf-runner
    |
    +-- EvaluationWorkload
    |     +-- case/unit admission
    |     +-- bounded host-operation arbitration
    |     +-- cancellation and exact call ledger
    |     `-- canonical report join
    |
    +-- RustEvaluationHost
    |     +-- inference -> prepared endpoint -> aiperf-transport-http
    |     +-- immutable assets -> Rust fetch/cache
    |     +-- sandbox/process capabilities -> Rust implementations
    |     +-- explicitly registered tool/resource capabilities
    |     `-- scoped local HTTP/SSE compatibility proxy -> same host executors
    |
    `-- supervised, network-isolated evaluator-provider worker
          +-- nemo_evaluator
          |     +-- seed/prompt/agent/environment/verifier logic
          |     `-- canonical NEL bundle
          `-- openbench
                +-- pinned Inspect task/solver/scorer/epoch logic
                `-- canonical Inspect EvalLog bundle

Host effects cross either typed correlated inherited pipes or a Rust-minted, per-run local proxy
route. Python receives no real endpoint or provider credential. It may receive a local proxy URL
and an ephemeral scoped capability grant; the isolation policy permits no other network target.
Rust parses upstream SSE and may emit a normalized local compatibility stream. Raw upstream SSE
never crosses into Python.
```

Static and agentic accuracy are not different host architectures. A static problem is a session
case that normally emits one inference operation before verification. An agentic problem is a
session case that may emit many inference, tool, sandbox, or verifier operations. Batch evaluators
are session execution units with explicitly different scheduling granularity.

The delivery is ours on every side. We will implement the neutral NeMo Evaluator seams in our
fork, the OpenBench/Inspect provider adapter and manifests in our OpenBench fork, the minimal
generic Inspect AI changes that its official extension points cannot express, and the Rust
host/provider seams in AIPerf. Upstream acceptance may reduce long-term fork cost, but it is not a
prerequisite and cannot be a correctness dependency.

---

## 1. Why the current boundaries are insufficient

### 1.1 AIPerf has two evaluator protocols for one domain

`aiperf-accuracy` currently exposes a one-shot `AccuracyEvaluator` (`load`, page problems,
`grade_batch`) and an inherited `AgenticEvaluator` (`load_agentic`, page/start/poll/submit/cancel/
finish) in `crates/aiperf-accuracy/src/worker.rs:137-241`. The static DTO publishes model-safe
prompts and later receives response text in `crates/aiperf-accuracy/src/protocol.rs:166-230`; the
agentic DTO publishes chat-shaped calls in `crates/aiperf-accuracy/src/protocol.rs:438-473`.

The stateful path is closer to the correct general contract. It already has opaque cases, bounded
polling, correlated calls, cancellation, finalization, and exact outstanding-call validation.
The static path is the one-call specialization of that lifecycle, not a reason to preserve a
second protocol.

The local working tree also proves that stateful execution is no longer merely a library concept:
`crates/aiperf-runner/src/registry.rs:581-588` registers the `agentic` workload and
`online_http + agentic` pair, and `crates/aiperf-runner/tests/agentic_process.rs:569-666` exercises
the runner subprocess. Any migration must preserve that working behavior even where older docs
still call the pair pending.

### 1.2 The current callback gateway is too narrow to be the target proxy

Primary AIPerf agent calls already use the useful pattern: Python's `ModelCallBroker` emits work
and never opens a socket (`src/aiperf/accuracy/model_broker.py:4-10`), while Rust dispatches it
through the ordinary scheduled path.

Environment and verifier calls take another route. `crates/aiperf/src/agentic_gateway.rs:131-186`
binds an HTTP server, and `crates/aiperf-accuracy/src/protocol.rs:352-364` sends its URL and bearer
credential to Python. Harbor then installs those values as model-client configuration
(`src/aiperf/accuracy/harbor.py:307-332`). The gateway's streaming response is a buffered,
synthetic SSE projection (`crates/aiperf/src/agentic_gateway.rs:514-587`).

Rust already hosts that server, so it is useful compatibility evidence. Its current split is still
not the target: primary calls use pipes while auxiliary calls use a chat-only callback; routing is
single-endpoint; streaming is buffered; and the gateway is not integrated with a general
host-operation registry. The target replaces this implementation with one scoped compatibility
proxy backed by the same route admission, transport, ledger, cancellation, and reporting as typed
pipe operations. Providers may choose the pipe host or proxy adapter per declared operation;
neither path reaches an upstream endpoint from Python.

### 1.3 NeMo Evaluator currently owns effects that AIPerf Rust must own

NeMo Evaluator has the correct semantic decomposition: `EvalEnvironment` owns seed and verify,
while `Solver` owns solving (`src/nemo_evaluator/environments/base.py:27-97` and
`src/nemo_evaluator/solvers/base.py:42-65`). Its current orchestration nevertheless constructs
concrete network clients and service URLs around that decomposition:

- `engine/model_client.py:77-124,149-206,257-393` owns an `aiohttp` session, authentication,
  concurrency, retry, caching, chat, tool, VLM, completion, and embedding traffic;
- `adapters/interceptors/endpoint.py:140-280` executes upstream HTTP and retry policy;
- `adapters/interceptors/streaming.py:36-63,91-153` parses/coalesces SSE in Python;
- `adapters/proxy.py:54-121,226-304` starts and probes an HTTP proxy;
- `orchestration/orchestrator.py:698-743,855-947` resolves model URLs/keys and constructs the
  clients and batch environments;
- networked Gym, tool, NAT, Harbor, OpenClaw, container, sandbox, exporter, and remote-dataset
  implementations open additional network paths.

None of those implementations may reach a real service in AIPerf mode. Pure client/codec portions
may be retained behind an audited proxy host only when bound to the Rust local proxy with Python
retry, cache, endpoint discovery, and upstream credential handling disabled. The unrestricted
implementations remain behind the standalone NeMo Evaluator host.

### 1.4 NeMo Evaluator's natural session is trapped inside one function

`run_evaluation` already owns the complete semantic lifecycle in
`engine/eval_loop.py:70-885`: selection/shuffle/sharding, environment preparation, concurrency,
case retry, seed/solve/verify, cleanup, and aggregation. The per-case implementation is a nested
`_run_step` closure (`engine/eval_loop.py:253-708`). Extracting a session and host interface from
that code is less risky than recreating its semantics in Rust. A local proxy can carry compatible
model traffic, but it does not expose session planning, case identity, assets, outcomes,
cancellation, aggregation, or artifacts and therefore cannot replace the session contract.

### 1.5 OpenBench is an Inspect task catalog, not a second engine

OpenBench delegates execution to pinned Inspect AI. Its CLI resolves OpenBench task factories and
calls `inspect_ai.eval` with model URLs, concurrency, retries, sample scheduling, epochs, sandbox,
and logging controls (`src/openbench/_cli/eval_command.py:727-841`). Inspect then owns sample/task
scheduling, solver execution, scoring, epoch reduction, and the canonical `EvalLog`.

That difference must not leak into AIPerf's protocol. `openbench` is a sibling provider whose
adapter implements the same provider session contract using OpenBench task semantics plus pinned
Inspect execution. It is not wrapped by NeMo Evaluator, and NEL is not wrapped as an Inspect task.

The clean model-call seam already exists. Inspect's public `ModelAPI.generate` contract accepts
messages, tools, tool choice, and generation configuration
(`/tmp/inspect-ai/src/inspect_ai/model/_model.py:127-233`), and its registry supports named custom
providers. An `aiperf/<logical-service>` `ModelAPI` can therefore emit typed pipe operations and
return normalized `ModelOutput` without a Python client of the real endpoint or an Inspect
monkeypatch. This is the preferred Inspect adapter. A declared compatibility adapter may instead
use an ordinary Inspect provider against the local Rust proxy when it preserves exact call
identity, retry/cache policy, and result semantics.

Inspect's public `sandboxenv` registry similarly permits a Rust-backed sandbox implementation
(`/tmp/inspect-ai/src/inspect_ai/util/_sandbox/registry.py:16-41`). These official seams are used
before considering a pinned Inspect change.

### 1.6 OpenBench also contains bypasses that must fail closed or move behind the host

The OpenBench CLI and task catalog currently contain paths that bypass an injected model provider:

- the CLI accepts `model_base_url`, model API retry/timeout controls, direct cache preparation,
  Docker sandbox selection, and Hugging Face export (`_cli/eval_command.py:308-662,736-883`);
- the registry mutates Inspect's private global registry to override Groq, OpenRouter, and vLLM
  providers (`_registry.py:198-249`);
- the Groq provider constructs `httpx`/Groq clients and parses streaming chunks in Python
  (`model/_providers/groq.py:62-208,312-460`);
- most dataset sources are not revision-pinned: the repository contains 52 `hf_dataset` call
  sites but only two explicit `revision=` arguments at the research baseline; other loaders use
  direct HTTP URLs, `urllib`, `httpx`, git, or Hub download helpers;
- many LLM scorers call hard-coded `get_model(...)` providers, for example SimpleQA
  (`scorers/simpleqa.py:98-122`) and HLE's direct OpenAI fallbacks (`scorers/hle.py:67-76`);
- LiveMCPBench supports MCP SSE as well as stdio (`tools/livemcpbench/copilot/mcp_connection.py:
  12-75`), while code agents and some sandboxes launch clients that may make their own network
  calls.

The AIPerf path does not invoke the OpenBench CLI and does not load its direct provider clients.
It uses a programmatic provider session, immutable asset bindings, explicit logical model roles,
and capability-audited task manifests. Unsupported task effects fail during planning.

### 1.7 Coverage and semantics are not yet equivalent

Immediate deletion is unsafe:

- NeMo Evaluator does not currently provide exact canonical equivalents for every AIPerf static
  benchmark, BrowserGym provider, or MCPMark provider;
- same-name benchmarks may differ in prompts, selection, versions, and scoring;
- NeMo Evaluator's `lm_eval` environment skips log-likelihood tasks
  (`environments/lm_eval.py:68-101`);
- NeMo Evaluator requires Harbor `>=0.3,<0.4`, while AIPerf pins Harbor `0.18.0`; the existing
  BrowserGym/AgentLab and MCPMark environments also have independent locks;
- NeMo Evaluator currently converts several infrastructure failures to reward zero and includes
  them in aggregation (`engine/eval_loop.py:383-609,805-821`), while current AIPerf stateful
  reporting separates completed, infrastructure, and cancelled outcomes.
- OpenBench canonical results are Inspect `Score` trees, not only floats: values may be scalar,
  sequence, or mapping (`/tmp/inspect-ai/src/inspect_ai/scorer/_metric.py:34-55,97-110`), and
  named reducers combine sample epochs;
- Inspect full `EvalLog` samples include target, messages, output, scores, metadata, store, events,
  and attachments (`/tmp/inspect-ai/src/inspect_ai/log/_log.py:262-367`), so they are restricted
  evaluator artifacts rather than report-safe DTOs;
- OpenBench's CLI group summary is presentation computed after independent task logs
  (`_cli/eval_command.py:184-305,845-849`), not a canonical per-task `EvalLog` result.

The architecture can converge immediately; providers move only when their evidence is green.

---

## 2. Alternatives considered

| Alternative | Network ownership | Unifies static/stateful | Migration safety | Verdict |
|---|---:|---:|---:|---|
| Use an unscoped OpenAI proxy as the whole integration | Partial | Superficially | Low | Rejected |
| Monkeypatch NEL, OpenBench, or Inspect internals | No enforceable boundary | No | Low | Rejected |
| Add NEL as today's `AgenticHarnessProvider` | Retains callback path | No | Medium | Bring-up experiment only |
| Nest OpenBench under NEL (or NEL under Inspect) | Ambiguous | Superficially | Low | Rejected |
| Replace AIPerf transport/scheduling with an evaluator | No | Yes | Low | Rejected |
| Delete all current providers immediately | Potentially | Yes | Very low | Rejected |
| Neutral provider session + typed Rust host + scoped MITM compatibility proxy, sibling providers, staged replacement | Yes | Yes | High | **Chosen** |

### 2.1 Why a proxy is an adapter, not the whole boundary

A scoped local proxy is allowed and is valuable for third-party model clients. Rust terminates the
local request, checks a Rust-minted grant, resolves a logical route, owns upstream HTTP/SSE, and
records the same call ledger as a pipe operation. It is not a lease of the real endpoint.

A proxy alone is insufficient: an unscoped OpenAI facade hides call purpose and case lineage,
permits duplicate Python retries/caches, and cannot express dataset, sandbox, browser, MCP, batch,
outcome, or artifact lifecycles safely. The provider session and typed host-capability plan remain
mandatory. Proxy routes are registered dialect adapters with explicit operations and grants, not
a generic forward URL or arbitrary raw-HTTP tunnel.

### 2.2 Why the existing agentic provider seam is not the target

It would force NeMo Evaluator through Harbor-shaped controls, a single endpoint, chat-only DTOs,
provider-prefix selection, and duplicate aggregation. Its callback gateway is useful proxy
evidence but not the public shape of the replacement.

### 2.3 What “full replacement” means

It means eventual deletion of AIPerf's Python benchmark, grader, and stateful provider
implementations after exact parity. It does **not** delete:

- Rust evaluator control DTOs and process supervision;
- the Rust case/call scheduler and `SlotPool` admission;
- `RequestSink`, `TurnDispatcher`, endpoint bindings, or the Clock;
- DNS/TCP/TLS/HTTP/SSE, retry, cancellation, or telemetry;
- Rust-authoritative inference accounting;
- native-v2 report assembly and artifact integrity checks.

---

## 3. Non-negotiable ownership invariants

### 3.1 Evaluation semantics belong to the selected provider

The selected provider owns:

- benchmark/provider resolution and immutable dataset semantics;
- selection, shuffle, shard, repeat, seed, and canonical case order;
- public prompt/message/tool construction and generation requirements;
- agent/solver state machines and semantic case retries;
- local tool semantics that require no external effect;
- hidden answers, private tests, verifier state, and scorer/judge policy;
- provider-native per-case score trees/rewards, aggregation, confidence intervals, epoch reducers,
  pass@k, categories, and the canonical evaluation bundle.

AIPerf does not reinterpret prompts, hidden data, rewards, or headline metrics.

For `nemo_evaluator`, NEL owns seed → solve → verify and its bundle. For `openbench`, OpenBench
task/dataset/scorer code plus the pinned Inspect runtime own sample/epoch execution, reductions,
metrics, and the canonical `EvalLog`. Provider identity is explicit; benchmark names never trigger
implicit cross-provider fallback.

### 3.2 All upstream network-bearing effects belong to AIPerf Rust

AIPerf Rust owns:

- DNS, TCP, UDP, Unix-domain network sockets, TLS, HTTP/1, HTTP/2, and SSE;
- model, environment-model, verifier-model, judge, simulator, reward-model, completion, Responses,
  embedding, and multimodal network calls;
- endpoint URLs, credentials, headers, connection reuse, readiness, transport retries, and
  response streaming/terminal assembly;
- remote immutable dataset/artifact acquisition;
- network-bearing sandbox/container/provider lifecycle operations;
- approved network-backed tools and resources through explicit typed implementations;
- network cancellation, timeouts, trace timing, usage reconciliation, and telemetry.

Python receives no real endpoint URL, upstream credential, Docker/provider socket, signed asset
location, or temporary endpoint lease. A provider host may receive only a Rust-minted local proxy
locator plus an ephemeral capability grant bound to declared logical routes, operation schemas,
case context, and budgets. The proxy never accepts a caller-chosen upstream URL and is not a
generic raw-HTTP tunnel. Semantic URL text in benchmark content is inert; dereferencing it is a
separate typed Rust-owned asset/resource effect.

### 3.3 The evaluator process is isolated; network is limited to the Rust proxy

Every AIPerf evaluator-provider worker and all descendants run under one mandatory
`EvaluatorIsolation` policy. Its independently replaceable filesystem, environment, process, and
network implementations are selected by the registered provider factory; `NetworkIsolation` is a
leaf of that larger boundary, not the complete sandbox. The initial Linux profile provides:

- a cleared, allowlisted environment containing no host or upstream secret;
- a private mount/root view exposing only the worker distribution, declared read-only assets, and
  its contained writable staging directory;
- a dedicated unprivileged identity, a restricted or private `/proc`, `no_new_privs`, and bounded
  CPU, memory, file, process, and descriptor resources;
- no unexpected inherited descriptor or host service, including Docker/provider sockets; and
- a network policy permitting only explicitly granted connections to the per-run Rust
  compatibility proxy. Direct DNS, Internet, host-network, and peer-worker access is denied.

Providers using only the pipe host receive no network grant at all. A proxy grant is scoped to its
declared process subtree and logical routes; an ungranted child cannot inherit it accidentally.
The exact enforcement mechanisms remain trait-backed because platforms differ, but the observable
isolation contract does not. A distribution that cannot enforce the whole required profile does
not advertise the provider. The same rule applies to Inspect sandboxes and provider-launched
descendants; declaring only the top-level worker isolated is insufficient. A unit-test monkeypatch
is not an enforcement mechanism.

### 3.4 Public control state contains no hidden material; restricted inference may transit

Case descriptors contain opaque IDs and safe reporting labels only. Expected answers, hidden
tests, verifier state, judge references, private scorer reasoning, and restricted artifacts never
appear in public control metadata, primary-model context, diagnostics, request artifacts, or public
report fields.

When canonical scoring requires a remote judge/verifier to see a reference, the task manifest may
declare a purpose-tagged `RestrictedInferencePayload`. Its sensitive body may transit the typed
host operation or scoped Rust proxy solely to the granted judge/verifier route. Rust treats the
body as transient secret material: content logging, request capture, cache, public hashing,
cross-route reuse, and diagnostic echo are disabled. Only correlation, route, timing, usage, and
terminal disposition enter the public ledger. Undeclared disclosure fails closed. Rust may seal a
restricted provider artifact without publishing its contents.

`nemo_evaluator.serving.app` is explicitly not an integration surface: its seed response includes
`expected_answer` (`serving/app.py:32-75`). Remote Gym seed/verify is unsupported until a local
pipe-native form keeps hidden state inside the evaluator trust boundary.

### 3.5 AIPerf remains one product path

Users invoke the Python `aiperf` CLI. It resolves Config v2 and starts one fresh
`aiperf-runner`. The runner starts the evaluator worker as an internal supervised child. There is
no second human-facing NEL/OpenBench/Inspect CLI in an AIPerf run and no Python client fallback to
a real inference endpoint. A local client bound exclusively to the Rust compatibility proxy is an
internal host adapter, not another product path.

---

## 4. Neutral provider session and host contract

### 4.1 `EvaluationSession`

Every evaluator adapter implements the same neutral session API, conceptually:

```python
@dataclass(frozen=True)
class CaseTemplateDescriptor:
    template_id: str
    task: str
    source: str

@dataclass(frozen=True)
class ExecutionUnitTemplateDescriptor:
    unit_template_id: str
    case_template_ids: tuple[str, ...]
    granularity: Literal["case", "host_batch"]
    scheduling_class: str

@dataclass(frozen=True)
class CaseOccurrenceDescriptor:
    case_id: str
    template_id: str
    issue_ordinal: int
    phase_id: str
    cycle_index: int

@dataclass(frozen=True)
class ExecutionUnitOccurrence:
    unit_id: str
    unit_template_id: str
    cases: tuple[CaseOccurrenceDescriptor, ...]

class CaseOutcomeKind(StrEnum):
    COMPLETED = "completed"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    CANCELLED = "cancelled"

@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    kind: CaseOutcomeKind
    scores: Mapping[str, ProviderScore]
    numeric_metrics: Mapping[str, float]
    primary_score: str | None
    error: EvaluationError | None
    artifact_refs: tuple[ArtifactRef, ...]

class EvaluationSession(Protocol):
    identity: EvaluationIdentity
    case_templates: Sequence[CaseTemplateDescriptor]
    unit_templates: Sequence[ExecutionUnitTemplateDescriptor]
    requirements: HostRequirements
    aggregation_policy: AggregationPolicy
    scheduling_mode: Literal["finite", "rust_occurrences"]

    async def instantiate_units(
        self, requests: Sequence[UnitOccurrenceRequest]
    ) -> Sequence[ExecutionUnitOccurrence]: ...
    async def run_unit(self, unit_id: str, host: EvaluationHost) -> Sequence[CaseOutcome]: ...
    async def cancel_unit(self, unit_id: str) -> None: ...
    async def finalize(self) -> EvaluationBundle: ...
    async def close(self) -> None: ...
```

Required semantics:

1. `prepare` freezes selection and ordered templates before measured work. Template IDs bind
   provider case identity, provider-owned replicate/epoch, candidate, and provider revision.
2. A `finite` plan instantiates every occurrence during preparation and starts each exactly once.
   A `rust_occurrences` plan lets Rust instantiate an ordered occurrence when the normal phase
   scheduler issues work. Its case ID additionally binds phase, global issue ordinal, and cycle
   index. Instantiation is deterministic, idempotent, and cannot change template semantics.
3. Descriptors are model-safe and contain no prompt or truth material.
4. Completion order never changes canonical result order: finite plans use frozen provider order;
   scheduled plans use Rust issue ordinal after provider-owned template order.
5. The provider owns its complete semantic lifecycle for a unit. NEL owns seed → solve → verify;
   OpenBench/Inspect owns task → solver → scorer → epoch reducer.
6. A semantic retry creates a new attempt ID under the same case; it does not reuse a call ID.
7. Finalization rejects duplicate or missing terminal outcomes, aggregates once, writes the
   provider's canonical bundle, and returns a sealed manifest candidate.
8. `close` is idempotent and tears down every local semantic task even after failure.

Provider-native repeat remains semantic policy and is represented in templates; Rust cycle index
is only load-scheduler occurrence identity. NEL may keep all repeats for one source problem inside
one unit and seed exactly once. `host_batch` providers are initially `finite`. The old static
accuracy path may migrate to duration/request-rate phases only after its provider advertises
`rust_occurrences`; otherwise those phase shapes fail authored validation rather than silently
executing a finite plan once.

`ProviderScore.value` is bounded canonical JSON matching the provider's native score algebra; it
is not coerced to float. The complete value is restricted by default. `numeric_metrics` is the
separate finite projection eligible for AIPerf score/performance joins. A registered provider
descriptor may additionally define a versioned, reviewed public projection schema for fixed
finite scalars or bounded enums such as `C`/`I`/`P`/`N`; Rust parses and reserializes that
projection. This preserves NEL float rewards and Inspect scalar/list/mapping scores without moving
reducer policy into Rust or trusting arbitrary provider-authored strings as public data.

The finalized bundle separately returns named `AggregateMetric` values with provider scorer,
reducer, metric name, finite numeric value, scored count, and unscored/excluded count. Provider
score explanations, answers, history, metadata, errors, annotations, and artifact bytes remain
restricted unless the registered Rust factory owns an explicit public projection schema. Provider
visibility labels are requests, not report authority; unknown or unregistered projections fail
closed to restricted.

Execution granularity is explicit. NEL's `EvalEnvironment.run_batch()` can own a whole loop
(`environments/base.py:93-97`). OpenBench's public Inspect API owns one task batch and does not
expose a public externally scheduled per-sample runner. A batch is therefore a declared unit, not
a fake giant case. An AIPerf host rejects a batch mode that cannot expose exact case/call outcome,
traffic accounting, and whole-unit cancellation.

### 4.2 `EvaluationHost`

Provider cores/adapters depend on narrow typed protocols:

```python
class InferenceHost(Protocol):
    async def infer(self, request: InferenceRequest, context: CallContext) -> InferenceResult: ...
    def stream(self, request: InferenceRequest, context: CallContext) -> AsyncIterator[InferenceEvent]: ...

class AssetHost(Protocol):
    async def acquire(self, request: PinnedAssetRequest) -> ResolvedAsset: ...

class SandboxHost(Protocol):
    async def create(self, request: SandboxCreate) -> SandboxHandle: ...
    async def execute(self, request: SandboxExecute) -> SandboxResult: ...
    async def destroy(self, handle: SandboxHandle) -> None: ...

class ProcessHost(Protocol):
    async def start(self, request: ProcessStart) -> ProcessHandle: ...
    async def interact(self, request: ProcessInteraction) -> ProcessResult: ...
    async def stop(self, handle: ProcessHandle) -> None: ...

class ResourceHost(Protocol):
    async def invoke(self, request: ResourceOperation) -> ResourceResult: ...

class EvaluationHost(
    InferenceHost, AssetHost, SandboxHost, ProcessHost, ResourceHost, Protocol
):
    identity: HostIdentity
    capabilities: HostCapabilities
```

Every interface is replaceable. A provider's local/default host preserves its standalone behavior
by wrapping today's Python network/process implementations. `PipeEvaluationHost` contains no
network, URL, credential, subprocess-provider, or Docker logic; it correlates typed operations
with the supervised worker protocol.

`ProxyEvaluationHost` is a compatibility implementation of the same declared capabilities. It
binds provider clients to one local Rust proxy locator and injects short-lived Rust-minted grants
plus correlation metadata. The Rust proxy validates and strips that internal metadata, lowers the
request through the registered host executor, and returns terminal JSON or a normalized local SSE
projection. It never gives the provider a real endpoint or upstream credential.

There is intentionally no upstream `HttpHost`, `EndpointLease`, caller-selected forward URL, or
arbitrary shell/network escape hatch. A new networked evaluator effect requires a named DTO or
registered proxy dialect, a capability descriptor, a Rust executor trait implementation, and
conformance tests. Pipe and proxy implementations must produce the same operation ledger.

### 4.3 Inference operation identity is open but not arbitrary

`InferenceRequest` carries:

- immutable session, unit, case, semantic-attempt, and logical-call IDs;
- a logical `service_id` and purpose (`primary`, `environment`, `verifier`, `judge`, `simulator`,
  `reward_model`, `user`, `base`, or provider-defined reporting label);
- an open validated semantic operation ID such as `model.generate`, `model.complete`,
  `model.responses`, or `model.embed`;
- a schema-validated operation payload;
- requested terminal or true-streaming response mode;
- deadline/idempotency hints, not transport implementation or retry counts.

Operation IDs are registered with descriptors and factories. They are not a closed Rust enum and
not an HTTP method/path supplied by Python. A descriptor states input/output schemas, modality,
streaming, usage, and endpoint compatibility. The runner fails preparation when a provider's
required operation has no executable Rust adapter for its route.

Rust parses upstream SSE. A pipe host receives typed `InferenceEvent` deltas/usage/terminal events;
a proxy host may receive a Rust-generated local SSE compatibility stream. No provider receives raw
upstream SSE bytes. A buffered terminal projection cannot claim true-streaming capability.
Inspect's `ModelAPI.generate` is terminal, so the preferred OpenBench pipe provider consumes the
Rust terminal result while retaining all upstream SSE timing solely in Rust.

### 4.4 NeMo Evaluator provider adapter

Solvers should not all be rewritten around AIPerf. `ModelClient` becomes a pure request-builder
and response-adapter over `InferenceHost`, retaining the solver-facing API where sensible. It loses
base URL, API key, `aiohttp`, semaphore, retry sleeps, and response cache state. CompletionSolver's
use of private `_post_with_retry` (`solvers/completion.py:44-106`) is replaced by a public typed
completion operation.

Standalone behavior moves to `hosts/local/inference.py`; AIPerf behavior terminates in
`PipeEvaluationHost` or, for an audited compatibility client, `ProxyEvaluationHost`.

### 4.5 NEL compatibility wrapper

`run_evaluation()` remains the standalone entry point. It constructs `LocalEvaluationHost`, a
local scheduler, and the extracted `EvaluationSession`. Existing `nel eval run` behavior is kept
as a compatibility target. The evaluation core cannot import local host implementations.

### 4.6 OpenBench/Inspect provider adapter

One `openbench` provider session contains exactly one exact OpenBench task, one candidate logical
service, zero or more declared auxiliary logical services, and one frozen sample/epoch selection.
The initial product surface rejects OpenBench group names and multi-candidate lists rather than
importing OpenBench into the Python frontend. Users author one task/candidate run at a time. A
future provider-neutral suite-planning runner operation may resolve a provider-owned group into
opaque exact-task run specs that Python only launches; it still may not move task resolution or
suite aggregation into Python. One measured runner is not an alternate implementation of
OpenBench's multi-task presentation loop.

The initial native granularity is `host_batch`: pinned Inspect runs the frozen task batch and emits
correlated effects tagged by `(case_id, epoch, semantic_attempt, call_ordinal, service_id)`. The
provider still returns safe per-case outcomes and canonical epoch reductions. Rust admits and
dispatches every effect and can cancel the batch; it does not call Inspect's private
`task_run_sample` (`inspect_ai/_eval/task/run.py:548+`). Per-case Rust-started execution is a future
capability that requires a public Inspect sample-session lifecycle with exact setup, cleanup,
reducers, and cancellation.

The adapter adds a lazy, explicit runtime rather than importing OpenBench's all-task registry:

```text
openbench/runtime/contracts.py
openbench/runtime/host.py
openbench/runtime/inspect_session.py
openbench/runtime/capabilities.py
openbench/runtime/artifacts.py
openbench/model/_providers/aiperf_pipe.py
openbench/sandbox/aiperf_pipe.py
openbench/resources/manifest.py
```

It invokes `inspect_ai.eval_async` programmatically with task and `Model` objects. It does not
invoke `bench eval`, patch display/recorders, import network provider implementations, export to
the Hub, or accept a model base URL.

### 4.7 Official Inspect extension points first

`AiperfPipeModelAPI(ModelAPI)` uses Inspect's public provider contract and registry. It:

- rejects `base_url`, API keys, credential/provider-client arguments, and provider batching;
- identifies only a logical service such as `aiperf/candidate` or `aiperf/grader`;
- converts Inspect messages, tools, tool choice, content, and generation configuration into one
  schema-validated inference operation;
- awaits the correlated terminal result over inherited pipes;
- returns a canonical Inspect `ModelOutput`/sanitized `ModelCall`;
- returns `false` from `should_retry`; and
- constructs no HTTP client and parses no stream.

The public `ModelAPI.generate` signature does not carry sample identity. The pinned runtime
therefore exposes a provider-neutral, read-only `ModelCallContext.current()` backed by task-local
context, not a private `TaskState` import. Inspect sets it around solver and scorer execution with
eval/task identity, sample ID, epoch, semantic-attempt identity, and a per-sample monotonic call
ordinal. `AiperfPipeModelAPI` requires that context and combines it with the logical service role;
an absent, stale, or reused context fails before a host operation is emitted. Concurrent samples
cannot share counters or context.

Inspect normalizes message/tool/reasoning history before calling `ModelAPI.generate`
(`inspect_ai/model/_model.py:577-651`). The operation schema must losslessly cover its text,
reasoning, image, audio, video, document/data, tool, structured-output, multiple-choice, logprob,
usage, and stop-reason algebra. Unsupported fields fail capability negotiation; they are never
dropped.

`AiperfPipeSandboxEnvironment(SandboxEnvironment)` uses Inspect's public `sandboxenv` registry and
turns its methods into typed Rust sandbox operations. Audited tool/resource adapters use the same
approach. Official extension points are preferred because they remain ordinary Inspect APIs and
need no private registry mutation.

### 4.8 No-monkeypatch and minimal pinned-Inspect patch policy

The AIPerf path must not use OpenBench's current private/global mutations:

- `_registry.py:198-249` overwrites Inspect's private provider registry;
- `model/_providers/openrouter.py:128-145` replaces an instantiated client's `create` method;
- `_cli/eval_command.py:760-783` patches file-recorder and display behavior.

Ordinary model and sandbox dispatch requires **no private Inspect patch**: the official `ModelAPI`
and `sandboxenv` contracts are the integration seams. OpenBench datasets, logical roles, direct
SDK clients, and task capability metadata are fixed in our OpenBench fork. The three small generic
public APIs below cover fail-closed behavior the official seams do not currently expose.

Where Inspect has no public fail-closed seam, we make a small generic commit in our pinned Inspect
fork rather than monkeypatching a private symbol. The first required addition is:

```python
class ModelAPI:
    def allows_cache(self) -> bool:
        return True
```

Inspect checks this before `cache_fetch` in `model/_model.py:666-699`; the pipe provider returns
`False`, making any `cache=True` call fail before it can bypass Rust. The current OpenBench tree
contains no `cache=True` model call, so the initial audited task subset can land before this patch,
but arbitrary-task support cannot be advertised without the fail-closed API.

The second required addition is a public, fail-closed entry-point loading policy. Integrated
`eval_async` runs select `entry_point_policy="deny"`: registry lookup and hook discovery may use
objects explicitly constructed or registered by the frozen runtime, but `ensure_entry_points()`
may not enumerate or load installed `inspect_ai` entry points. This prevents OpenBench's installed
entry point from importing `_registry.py`, mutating Inspect's private registry, eagerly importing
all tasks, or triggering import-time effects. A manifest-declared allowlist may be added later, but
warning-only entry-point failures are never accepted in reproducible mode.

The third required addition is the public `ModelCallContext` described above. Its defaults preserve
standalone behavior; a provider may require context and fail closed without changing every
`ModelAPI.generate` signature.

If future case-level Rust admission/cancellation is required, the next Inspect change is a public,
provider-neutral `SampleController`/session lifecycle. We do not implement it through fail-open
hooks or private `task_run_sample`. Inspect hooks explicitly swallow ordinary hook exceptions
(`inspect_ai/hooks/_hooks.py:186-192,508-518`), so they are observability only unless Inspect adds
a documented required/fail-closed controller contract.

These are clean, independently tested, upstreamable commits with generic names and defaults that
preserve standalone Inspect behavior. They are maintained in our pinned fork until accepted
upstream. No upstream submission or PR is part of this RFC execution.

### 4.9 OpenBench task manifests and logical roles

Planning must not import a task that downloads data or initializes a provider. Each advertised
task has a static manifest declaring:

- exact task factory/source identity and supported arguments;
- immutable asset IDs, revisions, content digests, and local binding names;
- one candidate role and every auxiliary role (`grader`, `user`, `base`, `embedding`, or another
  explicit logical service);
- required model operations/content types;
- sandbox/process/tool/resource capabilities;
- session granularity, epochs, reducers, and failure policy;
- whether the task is safe under the inherited process-tree isolation and scoped-proxy-grant policy.

The integrated loader imports only the exact task module named by that frozen manifest and passes
the resulting task through the programmatic runtime. It does not resolve the installed
`openbench` Inspect entry point or call the eager all-task registry. A subprocess test installs the
ordinary OpenBench entry point with a load sentinel and proves that `plan_session` and `eval_async`
never invoke it.

After Rust binds assets, an OpenBench `AssetResolver` gives task/dataset code verified local paths.
Standalone mode retains today's default URL/Hugging Face behavior; AIPerf mode rejects an
unresolved or mutable asset. This requires refactoring direct URL/download call sites rather than
pre-populating an undocumented cache layout.

Hard-coded model construction becomes a declared role. SimpleQA, HLE, HealthBench, tau-bench,
MMStar, and the other LLM scorers/solvers may keep their standalone model defaults, but AIPerf
mode requires a pipe- or scoped-proxy-backed logical role and forbids provider fallback after
failure.

### 4.10 Inspect scheduling, retry, cache, and result rules

For the AIPerf OpenBench provider:

- `GenerateConfig.batch` is false;
- Inspect model `attempt_timeout` is absent;
- `AiperfPipeModelAPI.should_retry` is false;
- sample `retry_on_error` is zero unless a future semantic-retry protocol assigns a new attempt;
- proxy-compatible provider-client retry is disabled; any duplicate local request with the same
  logical operation/idempotency identity is deduplicated by Rust and cannot create a second
  upstream logical call;
- Inspect model/sample semaphores bound only Python producer work and pipe pressure; Rust route
  `SlotPool`s are the only inference/network admission authority;
- `max_sandboxes=None`, and `AiperfPipeSandboxEnvironment.default_concurrency()` returns `None`.
  Inspect therefore does not enter its public random-delay branch, with no pinned-Inspect jitter
  patch or private scheduler hook;
- `eval_async` uses `fail_on_error=False`, `continue_on_fail=True`, `retry_on_error=0`, and complete
  sample logging. The adapter joins results back to the frozen `(sample_id, epoch)` manifest and
  emits exactly one typed terminal for every selected case. A missing/failed sample becomes
  `InfrastructureError`; a batch-level exception is run infrastructure and cannot silently omit
  cases or publish partial aggregates;
- model cache is forbidden in measured mode;
- cancellation resolves every pipe future, cancels Rust HTTP/SSE, and terminates the whole
  `host_batch` without publishing a score for unfinished cases.

Inspect stores `(sample_id, epoch)` and reduces epochs by sample identity. OpenBench case identity
therefore binds task instance, candidate, sample ID, and epoch. The provider preserves Inspect's
raw typed `Score` values and named reducers. Only sanitized finite aggregate metrics cross into
the public AIPerf report; the full `EvalLog`, which contains targets and rich sample state, remains
a restricted sealed artifact.

---

## 5. Supervised evaluator protocol v2

This is the **evaluator-worker protocol**, distinct from outer runner protocol v2. Both versions
must always be named in errors and reports.

### 5.1 Transport and framing

- Rust owns the child process and creates dedicated one-way control descriptors; conventional
  stdin/stdout are never the evaluator protocol. The minimal worker bootstrap receives only those
  fixed descriptors, marks them close-on-exec before importing provider code, and reserves them at
  file-descriptor level. Diagnostics use a separate supervised stderr channel; ordinary stdout is
  redirected away from protocol parsing.
- Every provider/tool/MCP/process child launch closes the master control descriptors and all proxy
  grant descriptors outside that child's declared scope before exec. Rust-hosted children receive
  only operation-scoped pipes/handles, never the evaluator control channel. A descendant cannot
  read, steal, or impersonate a control reply.
- Messages are strict, versioned JSONL with bounded line and collection sizes; large artifacts are
  referenced by contained staging paths, never inlined.
- Every request/reply has a correlation ID. Every semantic event has a monotonically increasing
  session sequence and an idempotency key.
- The worker never emits unsolicited stdout. Background session tasks enqueue events; Rust drains
  them with bounded long-poll operations.
- Unknown fields, operations, capabilities, IDs, duplicate terminals, late results, and sequence
  regressions fail closed.

### 5.2 Operations

```text
hello
plan_session
bind_assets
next_units
instantiate_units
start_units
poll_events
submit_host_events
cancel_units
finalize_session
shutdown
```

`plan_session` validates evaluator-owned configuration without model traffic or sandbox creation.
It returns evaluator identity, immutable asset requirements, host capability requirements,
logical service requirements, aggregation policy, and execution granularity.

Rust resolves immutable assets and executable capabilities, then `bind_assets` creates the frozen
template/unit plan. Planning may be split internally, but no model-safe template is considered
frozen until all identities and digests are known. For a declared `rust_occurrences` plan,
`instantiate_units` materializes deterministic occurrence IDs from Rust-authored phase, issue, and
cycle identity before `start_units`; a finite plan rejects that operation.

`poll_events` returns a bounded mixture of:

- `HostOperationRequested`;
- `HostOperationCancelRequested`, keyed by the original operation and semantic-attempt IDs;
- typed stream/credit state;
- `CaseTerminal`;
- evaluator progress and non-secret diagnostics.

`submit_host_events` sends zero or more typed deltas and exactly one terminal result per host
operation. Rust can therefore feed real streaming incrementally over pipes. A declared proxy host
drives the same operation ledger through its scoped local request lifecycle instead.

`HostOperationCancelRequested` is idempotent and races safely with normal terminal completion.
Rust stops queued or active transport, then returns exactly one cancelled or already-terminal
acknowledgement through `submit_host_events`. Closing a provider stream iterator and disconnecting
a scoped proxy request lower to this same event; neither may merely abandon a Rust operation.

### 5.3 State machine

```text
spawned
  -> negotiated
  -> planned
  -> assets_bound
  -> ready
  -> running <-> cancelling
  -> drained
  -> manifest_candidate
  -> quiescing
  -> worker_exited
  -> artifacts_sealed
  -> report_committed
```

No state may be skipped. `finalize_session` requires no running unit and no unresolved host
operation and returns only a manifest candidate. Rust then requests graceful shutdown, revokes
every sandbox/process/artifact-writing capability, and verifies that the worker and its complete
descendant tree have exited before artifact sealing begins. Cancellation is idempotent. Worker
crash, protocol corruption, identity drift, quiescence failure, or another invariant failure is
run infrastructure failure, never a wrong answer.

### 5.4 Backpressure and fairness

The protocol advertises queue credits. Rust bounds:

- started evaluation units;
- outstanding host operations globally and per unit;
- buffered stream events and maximum message bytes;
- sandbox/process handles;
- artifact count/bytes.

Rust arbitrates ready host operations fairly across units before admission to route-specific
`SlotPool`s. A single agent cannot monopolize the run by recursively emitting calls. The selected
provider blocks on the corresponding future when credits are exhausted. `start_units` only
acknowledges creation of background unit tasks; it never synchronously waits for those tasks, so
Rust can continue polling and satisfying their host operations without a pipe deadlock.

### 5.5 No-secret wire rules

Wire validation rejects upstream connection-authority and real-credential fields in evaluator
control DTOs. It permits inert URL-shaped benchmark content, the Rust-issued local proxy binding,
and manifest-declared restricted judge/verifier bodies. Logical service IDs map to Rust-owned
routes. Error messages redact resolved endpoints, headers, local grants, tokens, signed paths,
restricted bodies, and provider secrets. Sentinels prove that hidden material never appears in
case descriptors, ordinary diagnostics, primary-route calls, request artifacts, or public output;
separate taint tests prove declared restricted bodies reach only their granted auxiliary route.

---

## 6. AIPerf Rust boundary

### 6.1 Replaceable evaluator provider

`aiperf-accuracy` keeps its role as an IO-free, transport-free evaluator control crate but
generalizes its object-safe seam:

```rust
#[async_trait(?Send)]
pub trait EvaluationProvider {
    fn identity(&self) -> &EvaluationWorkerIdentity;
    async fn plan(&mut self, request: &EvaluationPlanRequest) -> Result<EvaluationPlan>;
    async fn bind_assets(&mut self, assets: &[ResolvedEvaluationAsset])
        -> Result<EvaluationIdentity>;
    async fn next_units(&mut self, offset: usize, limit: usize) -> Result<EvaluationUnitPage>;
    async fn instantiate_units(&mut self, requests: &[EvaluationUnitOccurrenceRequest])
        -> Result<Vec<EvaluationUnitOccurrence>>;
    async fn start_units(&mut self, ids: &[EvaluationUnitId]) -> Result<()>;
    async fn poll_events(&mut self, limit: usize, wait_ms: u64)
        -> Result<EvaluationEventBatch>;
    async fn submit_host_events(&mut self, events: &[HostOperationEvent]) -> Result<()>;
    async fn cancel_units(&mut self, ids: &[EvaluationUnitId]) -> Result<()>;
    async fn finalize_candidate(&mut self) -> Result<EvaluationFinishCandidate>;
    async fn shutdown(&mut self) -> Result<()>;
}
```

An object-safe `EvaluationProviderFactory` and deterministic open provider registry make every
evaluator replaceable; the runner never branches on a provider string. The stock factories are
`nemo_evaluator` and `openbench`. A provider descriptor declares worker protocol versions,
execution granularities, operation schemas, isolation requirements, identity fields, a versioned
authored-config schema/fingerprint, and a factory-owned launch identity.

A stock factory, not authored run configuration, selects the executable, worker module, literal
argv, clean environment closure, working directory, isolation profile, and expected source/lock or
OCI digest. A deployment may register multiple immutable distribution IDs, but a run can only
select one of those registered IDs; it cannot supply a program or import target. Before exec, Rust
independently hashes or verifies the selected closure. After `hello`, it compares the negotiated
identity with that evidence. Worker-reported versions and hashes are descriptive and cannot attest
the worker that authored them.

The descriptor's pure Rust validator strictly decodes provider-authored configuration during
runner protocol-v2 `validate`, without spawning Python, importing a provider, opening a sandbox,
or resolving an asset. `execute` repeats that exact validation, binds its schema fingerprint into
evaluation identity, and only then launches the worker. `plan_session` performs dynamic
provider-owned preparation against the same validated value; it cannot retroactively broaden the
authored schema. Distribution and schema drift fail closed.

The provider registry is not a Rust benchmark/grader registry. Benchmarks and graders stay inside
the selected evaluator. It registers only process-protocol implementations.

### 6.2 Trait-backed Rust host executors

The runtime defines object-safe host executor/factory seams. Each registered executor owns one
typed operation family and advertises a schema/capability descriptor. The stock inference executor
lowers through prepared endpoint bindings and the ordinary `RequestSink`/`TurnDispatcher` path.

Representative families:

- `InferenceHostExecutor`;
- `CompatibilityProxyIngress`, which authenticates local dialect requests and lowers them into
  `InferenceHostExecutor`/other explicitly registered executors;
- `AssetHostExecutor`;
- `SandboxHostExecutor`;
- `ProcessHostExecutor`;
- `McpHostExecutor` for audited stdio MCP only;
- future `BrowserHostExecutor` only when browser network effects are truly Rust-owned;
- explicitly named resource/tool executors.

Registering a descriptor is not enough. Runner capabilities publish a provider/backend/workload/
host-operation combination only when an executable adapter and subprocess proof are linked.

### 6.3 `EvaluationWorkload`

`crates/aiperf/src/evaluation.rs` is extracted from the reusable parts of current `agentic.rs` and
`accuracy.rs`. It owns:

- separate unit/environment concurrency and inference-route concurrency;
- canonical unit admission with `SlotPool`;
- the event poller and bounded fair host-operation queues;
- exact case/attempt/call ledgers;
- lowering inference operations through prepared route bindings;
- cancellation of queued, dispatched, streaming, and verifier-stage work;
- validation of every terminal and final canonical order;
- joins between evaluator case outcomes and Rust performance records.

It does not build a benchmark prompt, execute a hidden test, compute a reward, or aggregate the
canonical score.

### 6.4 Logical routes and multiple services

Protocol-v2 evaluation configuration owns an explicit route table:

```yaml
workload:
  type: evaluation
  config:
    provider:
      type: nemo_evaluator
      distribution: nvidia_nemo_evaluator_0_4_locked
    evaluation:
      benchmark: "..."
      # NeMo-Evaluator-owned, schema-validated provider config
    routes:
      primary:
        model: candidate
        endpoint_profile: candidate_openai
      judge:
        model: judge
        endpoint_profile: judge_anthropic
      embeddings:
        model: embedder
        endpoint_profile: embedding_openai
    resources: {}
    unit_concurrency: 8
```

Python Config v2 treats provider/evaluation-specific objects as opaque after structural checks.
The exact runner distribution strictly validates provider config, route references, endpoint
capability compatibility, resource declarations, and side-effect-free authored semantics through
the selected factory's fingerprinted schema. Executable/module/environment coordinates are not
authored fields.

The provider requests `service_id=judge`; it does not choose a URL. Rust resolves the route to model,
endpoint profile, prepared endpoint binding, credentials, admission pool, retry policy, and
metrics labels. Current `agentic_execution.rs:538-559` requires one model/default endpoint; the
generic workload removes that restriction only after multi-route execution tests exist.

### 6.5 Rust remains accounting authority

The evaluator supplies IDs and semantic results, not trusted network counts. Rust records:

- logical operation count;
- every transport attempt and retry lineage;
- route, service, purpose, model, endpoint profile, operation ID, and case/attempt correlation;
- arrival, admission, DNS/TCP/TLS/send/SSE/token/terminal timing;
- prompt, completion, reasoning, cached, and provider usage;
- completion, failure, and cancellation status.

Evaluator-authored token/call totals are removed from the authoritative report. A provider may
include trajectory annotations, but reconciliation is against Rust's call ledger.

---

## 7. Scheduling, retry, cache, and cancellation semantics

### 7.1 One owner at each layer

- The selected provider freezes canonical case order and owns semantic case flow.
- Rust decides when a unit starts and when each host operation is admitted.
- Rust alone retries a network attempt under the selected route policy.
- The provider may request a whole-case semantic retry only by opening a new semantic attempt with new
  operation IDs.
- Neither layer silently repeats the other's unit of work.

NEL's current `ModelClient`, endpoint interceptor, and evaluation loop can all retry
(`engine/model_client.py:332-393`, `adapters/interceptors/endpoint.py:140-250`, and
`engine/eval_loop.py:324-580`). AIPerf mode disables the first two; the last becomes explicit
semantic-attempt lineage.

Inspect's provider retry, sample retry, and batch-generation controls are likewise disabled as
specified in section 4.10. Inspect may retain task-level semantic retry only after it emits a new
semantic-attempt identity; it never retries a pipe model effect behind Rust's ledger.

`TransportRetryPolicy` is a replaceable Rust seam evaluated by an `InferenceAttemptExecutor` above
the one-attempt `RequestSink`. It classifies retryable terminals, allocates a fresh transport
attempt ID under one logical operation, sleeps only through the injected `Clock`, and records every
attempt. Cancellation interrupts dispatch and backoff. Retry after any externally observed output
is forbidden unless the registered operation and endpoint policy explicitly prove replay safety;
deadlines and idempotency are bound to the logical operation. Pipe and proxy clients do not own
upstream retry. Duplicate local proxy requests are deduplicated by logical operation ID.

### 7.2 Cache and resume

Measured runs default to no response cache and no verified-result resume. A cache/replay feature,
if later enabled, is a distinct declared run policy:

- every hit is returned by a Rust-owned cache or a fully identity-bound local evaluation cache;
- the report records source run, key schema, hit/miss, and excluded performance rows;
- replayed calls never masquerade as zero-latency model traffic;
- a cache key binds evaluator/provider source, full resolved config, dataset content, route model/
  endpoint identity, operation payload, adapter config, and retry policy.

NEL's current response cache and step-log hashes are not sufficient for measured AIPerf reuse
(`engine/cache.py:43-151`, `engine/step_log.py:44-58`). Inspect model cache is disabled by the
provider's fail-closed `allows_cache()` contract; reuse of a prior EvalLog is not a measured run.

### 7.3 Cancellation

Cancellation is correlated at run, unit, case, semantic-attempt, and host-operation levels.

1. Rust stops new unit and operation admission.
2. Queued operations receive one cancelled terminal without network dispatch.
3. Running inference uses ordinary AIPerf cancellation, including streaming teardown.
4. Provider-requested operation cancellation and proxy disconnects enter that same path; Rust
   sends one terminal acknowledgement for every outstanding operation.
5. The provider cancels solver/sample tasks, local tools, scoring/verification, and epoch work,
   then requests typed sandbox/process cleanup where applicable.
6. The worker acknowledges a drained unit. A bounded force-shutdown is infrastructure failure and
   cannot become a score.

No later phase/run starts while evaluator resources or host operations remain live.

---

## 8. Outcomes and aggregation

### 8.1 Typed terminal outcomes

```text
Completed(scores, primary_score, annotations)
InfrastructureError(stage, kind, retryable, message)
Cancelled(stage, reason)
```

A valid score of zero is `Completed`; it is never inferred to be infrastructure. Infrastructure
and cancellation carry no score map. Evaluator/system invariant failure aborts the session.

Provider policy must state how a failed inference terminal is handled. A static provider may
canonically grade a partial/empty response and produce `Completed(score=0)`, preserving today's
denominator semantics. A provider may instead classify a transport failure as infrastructure.
That decision is explicit in the frozen aggregation/failure policy, not imposed globally by Rust.

### 8.2 Canonical aggregation stays in the provider

The selected provider produces named per-case score trees plus canonical aggregate definitions
and finite numeric values. NEL retains its reward/headline aggregation; OpenBench retains Inspect
scorers, metrics, epoch reducers, and task results. Rust:

- validates bounded canonical JSON score values, finite public aggregate values, and case identity;
- stores generic score columns for performance/score joins;
- reports completed/infrastructure/cancelled counts separately;
- embeds a safe summary and sealed bundle digest;
- does not recompute pass@k, confidence intervals, category weighting, or headline choice.

The embedded AIPerf profile defaults to excluding infrastructure and cancellation from semantic
score denominators. A compatibility provider that needs another policy must declare it in identity
and prove parity before registration.

---

## 9. Assets, sandboxes, tools, and network-dependent providers

### 9.1 Immutable assets

Provider planning returns logical immutable asset requirements: source kind, immutable revision/content
digest, expected media type, and visibility. Rust resolves them before measurement through a
trait-backed `AssetHostExecutor`, using the existing Rust fetch/cache foundation where applicable.
Python libraries run in offline mode and receive contained read-only paths plus verified digests.
An unpinned or missing asset fails preparation.

This applies to benchmark datasets, task packages, model-independent images, OCI images, and
provider resources. It removes current direct Python Hugging Face/urllib acquisition paths.

### 9.2 Sandboxes and processes

Python receives opaque handles. Rust owns sandbox/provider API traffic and lifecycle. Sandbox
descriptors state filesystem mounts, environment allowlists, resource limits, image digest,
network mode, and cleanup policy. The default AIPerf network mode is `none`.

Spawning an arbitrary subprocess from Rust is not sufficient if that process can reach an
unapproved socket. The sandbox/process implementation must enforce the same egress policy and
mediate every allowed effect through an inherited typed pipe or explicit scoped proxy grant.

### 9.3 Tools and MCP

Local deterministic tools may run in an evaluator subtree with no proxy grant. External-effect tools are
typed `ResourceOperation`s implemented by Rust. A generic HTTP tool backend is prohibited.

MCP support starts with audited stdio servers under network denial. HTTP/SSE MCP transports are
unsupported. Each MCPMark service declares exact host requirements; filesystem-only services may
migrate before networked Notion, Playwright, GitHub, Supabase, or similar services.

### 9.4 Browser and remote-environment consequence

BrowserGym is not compliant today because the browser itself originates unrestricted network
traffic. It stays on the legacy provider until a Rust-owned browser/network capability exists and
passes the same audit. Remote Gym, NAT, opaque container batch, HTTP-only agents, and unscoped
callback-dependent Harbor/OpenClaw modes likewise fail preparation. A mode may migrate only when
all its external traffic terminates in registered Rust proxy/host operations and passes isolation.

This is deliberate capability truth, not a reason to weaken the boundary.

---

## 10. Provenance, artifacts, and trust

### 10.1 Evaluation identity

The frozen identity includes:

- evaluator protocol, selected provider package/version/source digest or commit, Python runtime,
  worker source digest, dependency lock digest, and optional OCI digest; `openbench` additionally
  binds its pinned Inspect version/commit and frozen task/plugin registry manifest;
- provider/plugin, environment, solver/agent, scorer/verifier/judge identities and source/package
  digests;
- secret-redacted fully resolved evaluator-config digest;
- dataset/task source, immutable revision/content digest, split, and ordered case/unit manifest
  digest;
- generation, adapter, selection, shuffle, shard, repeat, semantic retry, cache, and aggregation
  policies;
- Rust host identity, runner binary identity, host capability/schema inventory, and evaluator
  isolation proof;
- sandbox image/config digest and resource capability identities;
- secret-free logical route map and prepared endpoint identity digests.

Reproducible mode fails when a required identity is mutable or unknown. NeMo Evaluator's current
hand-maintained artifact/step hashes (`engine/artifacts.py:29-75` and
`engine/step_log.py:44-58`) and Inspect's partial package/Git provenance are supplemented by this
one canonical identity graph. OpenBench runtime entry-point overrides, warning-only plugin load
failures, mutable Hugging Face revisions, and process-randomized selection such as GPQA's
`hash(question)` are rejected until frozen or fixed.

### 10.2 Artifact staging and sealing

The runner assigns a contained staging directory. The provider writes only declared relative paths.
Finalization returns an artifact manifest with path, media type, visibility, size, and claimed
digest. Provider-authored artifacts are restricted by default; public eligibility requires a
factory-owned path/media/projection rule, and provider visibility labels cannot broaden it. After
`finalize_session`, Rust shuts down the worker, revokes host handles and staging access, and proves
the complete process tree quiescent. It then:

1. opens the staging root once and traverses with no-follow, FD-relative operations;
2. rejects absolute paths, traversal, hard-link escapes, symlinks, devices, undeclared files, and
   metadata changes during traversal;
3. hashes the opened size/content itself and verifies every claimed digest;
4. validates any public projection against its registered schema and reserializes it canonically;
5. atomically promotes an immutable Rust-owned tree;
6. writes the AIPerf report referencing the canonical bundle digest.

Restricted trajectories, hidden evaluation material, and private verifier artifacts are not
inlined in the public native-v2 report.

### 10.3 Report shape

Native v2 gains one generic `evaluation` block:

- evaluator/provider/host identity;
- safe resolved configuration and route inventory;
- ordered case outcomes with generic named scores;
- provider-native canonical aggregates and definitions;
- typed failure/exclusion counts;
- Rust-authoritative call/attempt/route summaries;
- sealed artifact manifest and canonical provider bundle digest.

For OpenBench, the full Inspect `.eval`/`EvalLog` stays restricted because samples may contain
targets, messages, outputs, arbitrary store/events, attachments, and scorer state. The public
projection contains only factory-schema-validated case labels and bounded score projections,
finite aggregate metrics, counts, identities, and the Rust-computed artifact digest. NEL result
rows and bundles containing expected answers are always restricted. Hub export is disabled.

### 10.4 Canonicalization and digest domains

`aiperf-canonical-json-v1` is the only cross-language semantic JSON codec. It rejects duplicate
keys, invalid Unicode scalar values, integers outside its declared signed/unsigned 64-bit domain,
non-finite floats, and unsupported values; preserves Unicode code points without normalization;
sorts object keys by UTF-8 bytes; and uses one specified shortest-round-trip IEEE-754 encoding
with negative zero normalized to zero and deterministic escaping. Rust and Python implementations
share byte-golden tests, and the codec version is part of identity.

Raw artifact digests hash exact file bytes and use `artifact_content_sha256`. Provider semantic
digests use the declared normalization followed by this codec and use
`normalized_result_sha256`. The domains and field names are never interchangeable.

Legacy `accuracy` and `agentic` blocks may be deterministic compatibility projections during one
migration window. They are not separate authorities.

---

## 11. Concrete change map: NeMo Evaluator fork

1. **Add `src/nemo_evaluator/engine/host.py`.** Define typed host subprotocols, IDs, contexts,
   operations/results, capability descriptors, and host identity. It imports no network/provider
   implementation.
2. **Add `src/nemo_evaluator/engine/session.py`.** Extract frozen selection, units, case ledger,
   finite/scheduled occurrence modes, typed outcomes, operation cancellation, and candidate
   finalization from `eval_loop.py`.
3. **Refactor `engine/eval_loop.py`.** Keep `run_evaluation()` as a compatibility wrapper around
   a session, local scheduler, and `LocalEvaluationHost`.
4. **Refactor `engine/model_client.py`.** Make it a pure facade over `InferenceHost`; move
   aiohttp/auth/retry/cache/semaphore behavior to the standalone local host.
5. **Replace `engine/model_call_context.py`.** Carry immutable session/unit/case/attempt/call,
   service, and purpose identity; remove URL rewriting.
6. **Split endpoint/streaming adapters.** Pure payload transforms stay in core. HTTP terminal
   transport and SSE parsing move under `hosts/local/`; the pipe host receives typed Rust events.
7. **Add `src/nemo_evaluator/hosts/local/`.** Wrap existing standalone inference, dataset,
   resource, process, and sandbox behavior without importing it into core.
8. **Add `src/nemo_evaluator/hosts/pipe.py` and an internal worker entry point.** Implement the
   bounded event/future broker and evaluator protocol v2 without network or serving code.
9. **Refactor solvers.** Chat/VLM/completion/react use injected typed clients. HTTP tool, Gym, NAT,
   Harbor, OpenClaw, and BYOB integrations either gain pipe-native adapters or declare unsupported
   requirements.
10. **Refactor environments/assets.** Dataset acquisition uses `AssetHost`; Gym keeps hidden seed
    state local; container environments declare batch granularity.
11. **Refactor sandbox abstractions.** Core talks to `SandboxHost`; Docker/Slurm/Apptainer/ECS
    implementations live only in standalone local hosts. AIPerf receives no Python Docker socket.
12. **Refactor `orchestration/orchestrator.py`.** Inject host/session factories. Standalone CLI
    constructs `LocalEvaluationHost`; AIPerf core receives no real endpoint or upstream key.
13. **Strengthen artifacts/schemas.** Add typed outcomes, canonical identity graph, complete
    config/case digests, visibility, and completed-only/default aggregation semantics.

Every step is a separately reviewable commit in our fork. The first commits are neutral library
refactors with standalone conformance tests; no AIPerf import enters NeMo Evaluator core.

---

## 12. Concrete change map: OpenBench and pinned Inspect

### 12.1 OpenBench fork

1. **Add `openbench/runtime/contracts.py` and `host.py`.** Define the provider-neutral session,
   host, identity, score, aggregate, outcome, and artifact contracts without importing AIPerf or
   any provider/network implementation.
2. **Add `openbench/runtime/inspect_session.py`.** Build one frozen task/candidate/epoch session,
   launch `inspect_ai.eval_async` in a background task, correlate effects to case/epoch/attempt
   IDs through public `ModelCallContext`, select the fail-closed error profile, project safe
   outcomes, preserve reducers, and finalize the native Inspect log.
3. **Add `openbench/model/_providers/aiperf_pipe.py`.** Implement the public Inspect `ModelAPI`
   seam with a logical service ID and typed inherited-pipe operations. Reject base URLs, keys,
   provider clients, cache, provider batch, internal retries, and unsupported content/config.
4. **Add `openbench/sandbox/aiperf_pipe.py`.** Implement the public Inspect `sandboxenv` seam as
   typed Rust sandbox operations. It creates no Docker client, socket, image pull, URL fetch, or
   subprocess with unmediated egress.
5. **Add `openbench/runtime/capabilities.py` and `resources/manifest.py`.** Use a frozen, lazy task
   registry with exact task source, arguments, assets, logical roles, operation/content schemas,
   reducers, granularity, sandboxes, network effects, and provider/plugin identities. Arbitrary
   entry points and warning-only plugin fallbacks are unavailable in AIPerf mode.
6. **Refactor dataset/resource acquisition.** An injected `AssetResolver` maps declared immutable
   IDs to Rust-bound read-only paths. Import-time/dynamic downloads, mutable Hugging Face defaults,
   Git clones, image pulls, and cache-layout assumptions are prohibited in the integrated path.
7. **Refactor model selection to logical roles.** Candidate, grader, user, base, embedding, and
   other explicit roles may retain standalone defaults, but AIPerf mode requires a declared
   pipe- or scoped-proxy-backed role and never falls back to another provider after an error.
8. **Audit direct effects task by task.** Google/Jina/Perspective calls, direct OpenAI/embedding
   clients, LiveMCP HTTP/SSE, networked code-agent CLIs, FactScore downloads, and similar paths
   either become named typed host operations or make the task fail capability negotiation.
9. **Fix failure classification and reproducibility.** Infrastructure caught as incorrect/zero
   becomes a typed infrastructure outcome; mutable inputs are pinned; unstable Python hashes use a
   stable digest; case selection and task registries receive golden manifests.
10. **Add `openbench/runtime/artifacts.py`.** Keep the complete `.eval` artifact restricted,
    return only safe projections and declared relative artifacts, and let Rust compute every
    promoted digest. The integrated path never runs Hub export.
11. **Factor pure run-spec construction from the CLI.** Standalone `bench eval` retains existing
    behavior and tests; AIPerf invokes the explicit programmatic session and none of the CLI's
    display, recorder, private-registry, or provider-client mutations.

### 12.2 Pinned Inspect fork

Official public extension points remain the default. The pinned Inspect fork receives only small,
provider-neutral additions that cannot be expressed through those points:

1. Add `ModelAPI.allows_cache() -> bool`, defaulting to `True`, and check it before cache lookup;
   the pipe provider returns `False`.
2. Add a public entry-point loading policy used by registry lookup and hook discovery. Standalone
   defaults to current discovery; AIPerf selects `deny` and uses only its explicitly frozen
   registrations.
3. Add public task-local `ModelCallContext`, populated around solver and scorer work and readable by
   custom `ModelAPI` implementations without importing private `TaskState` state.
4. Only if case-granular Rust admission becomes a requirement, add a public fail-closed
   `SampleController`/sample-session lifecycle covering setup, run, score, reduce, cancellation,
   and cleanup. Initial `host_batch` execution does not require or imitate this API.

Each addition has standalone-default, custom-provider, failure, and compatibility tests. No private
symbol replacement or runtime monkeypatch is accepted. No upstream submission or PR is authorized
by this RFC.

---

## 13. Concrete change map: AIPerf

### 13.1 `aiperf-accuracy`

- Add evaluator protocol-v2 DTOs and object-safe `EvaluationProvider`/
  `EvaluationProviderFactory` seams.
- Preserve strict JSONL correlation, line limits, stderr draining, identity validation,
  `kill_on_drop`, dedicated non-stdio control descriptors, and explicit shutdown/quiescence.
- Add generic case/unit/event/host-operation/outcome/identity/artifact types.
- Add template/occurrence DTOs and provider-requested host-operation cancellation.
- Add scoped local-proxy binding/grant DTOs as runner-to-worker preparation output, never authored
  upstream endpoint configuration.
- Remove `AgenticEvaluator: AccuracyEvaluator` inheritance in the new path.
- Delete `AgenticInferenceGatewayConfig` and the `agentic_inference_gateway` capability after the
  compatibility path retires.

### 13.2 `aiperf` runtime

- Add `crates/aiperf/src/evaluation.rs` from reusable `agentic.rs`/`accuracy.rs` scheduling,
  dispatch, ledger, cancellation, and report-join behavior.
- Add trait-backed Rust host executor composition and bounded fair arbitration.
- Add `TransportRetryPolicy`/`InferenceAttemptExecutor` over the one-attempt transport.
- Generalize `AgenticTurnBuilder` into an operation-aware evaluator materializer/router over
  prepared endpoint profiles.
- Route pipe and local-proxy ingress through one host-operation ledger, admission graph, transport,
  cancellation path, and reporter.
- Replace `crates/aiperf/src/agentic_gateway.rs`, its auxiliary queues, single-endpoint
  configuration, and buffered SSE with a trait-backed `EvaluatorCompatibilityProxy` after provider
  parity; do not retain the current gateway as a second authority.
- Retire the old `agentic.rs` and static `accuracy.rs` paths only after their deletion gates.

### 13.3 `aiperf-runner`

- Add `evaluation_execution.rs` and register the executable `online_http + evaluation` pair only
  with its process, isolation, routing, report, and subprocess proofs.
- Inject provider registry, isolation factory, asset resolver, route preparer, and host capability
  registry.
- Make provider factories own attested worker launches and fingerprinted, side-effect-free authored
  validators; accept no executable/module/environment coordinates from run config.
- Prepare multiple endpoint profiles worker-locally.
- Replace gateway factories, advertised-host probing, and callback capability checks with the
  isolation-scoped compatibility-proxy ingress factory and executable route/grant validation.
- Keep `static_accuracy` and `agentic` as temporary compatibility workload IDs; do not route them
  silently to incomplete new behavior.

### 13.4 Python frontend

- Project first-class evaluator provider, opaque evaluator config, logical routes, resources, and
  concurrency into runner protocol v2.
- Keep authored validation structural and side-effect-free.
- Do not import evaluator-provider packages, load a benchmark, fetch an asset, or contact a model from the
  Python CLI/orchestrator.

### 13.5 Provider code retired after parity

- in-tree static benchmark and grader implementations;
- `AgenticHarness`/`AgenticHarnessProvider` and dataset-prefix provider registry;
- AIPerf-specific Harbor, BrowserGym, and MCPMark providers;
- the MCPMark LiteLLM monkeypatch;
- Python model broker once its exact semantics live in the provider-neutral pipe or scoped-proxy
  host;
- Harbor-shaped agentic load/report configuration.

Isolated provider worker distributions remain valid. We do not merge incompatible Harbor,
BrowserGym, MCPMark, NeMo Evaluator, OpenBench, and Inspect dependencies into one Python
environment merely because the protocol is unified.

---

## 14. Implementation increments

### Increment 0 — executable boundary proof

- Define protocol/host/session DTOs and schema fingerprints in every participating repository.
- Start an isolated fixture worker with a pipe-only profile, then with one scoped proxy grant.
- Prove a clean environment/filesystem/process view and prove descendants cannot inherit or
  impersonate evaluator control descriptors.
- Prove correlated terminal and streaming inference over pipes and normalized local HTTP/SSE into
  the same ordinary Rust upstream HTTP stack and operation ledger.
- Prove syscall/namespace-level denial for every destination except the exact local proxy grant.
- Prove no real URL/upstream key/raw upstream SSE on either boundary and no restricted-body leak.

No real provider work starts before this increment is green.

### Increment 1 — neutral provider refactors

- In NeMo Evaluator, extract `EvaluationSession` and `LocalEvaluationHost`, preserve standalone
  selection/seed/solve/verify/aggregation/bundle behavior, and move direct networking behind local
  hosts.
- In OpenBench, add the explicit Inspect session, recording host, lazy frozen task registry,
  logical roles, safe result projection, and pipe `ModelAPI`/sandbox implementations while
  preserving standalone `bench eval` behavior.
- Use official Inspect extension points for ordinary calls; add no private patch or monkeypatch.

### Increment 2 — generic AIPerf evaluation workload

- Add the provider registry, process client, `EvaluationWorkload`, operation ledger, route table,
  generic report block, and `online_http + evaluation` pair.
- Add static template-to-occurrence scheduling and reject incompatible phase shapes until a
  provider advertises that mode.
- Support terminal chat first, then true streaming using the same typed event protocol.

### Increment 3 — immutable assets and simple static parity

- Resolve pinned datasets in Rust and bind read-only assets.
- Land the generic Inspect cache veto before advertising arbitrary OpenBench tasks; use the
  public `max_sandboxes=None` plus pipe-sandbox `default_concurrency() -> None` configuration before
  advertising sandbox tasks.
- Shadow selected pure static/chat tasks from each applicable provider with frozen responses and
  real canaries.
- Switch a provider only after byte-level prompt and result parity.

### Increment 4 — operation coverage and multi-route

- Add completions, Responses, embeddings, tools, multimodal, judges, and multiple logical routes as
  independently capability-gated adapters.
- Do not publish a capability before end-to-end subprocess proof.

### Increment 5 — Rust host effects

- Add sandbox/process and audited stdio MCP operations.
- Port pipe-native Harbor/tool integrations provider by provider.
- Keep browser, HTTP MCP, remote Gym/NAT, and opaque batch providers unsupported until they meet
  literal network ownership.

### Increment 6 — migration and deletion

- Run frozen-response differential tests, small real canaries, and shadow reports.
- Switch defaults one provider at a time with one compatibility release.
- Delete old providers/protocols/reports only when every applicable gate below is green.

These increments are a dependency order, not permission to create a PR or external repository
change. Repository publication requires a separate explicit instruction.

---

## 15. Acceptance and deletion gates

### 15.1 Boundary and security

- Isolation enforcement proves no evaluator worker or descendant reaches DNS, the Internet, host
  services, peers, or Docker/provider sockets; only an explicitly granted per-run Rust proxy
  target is reachable.
- Every external effect appears as a typed, capability-declared pipe operation or an authenticated
  registered proxy-dialect operation in the same Rust ledger.
- No endpoint lease, unscoped/generic proxy, Python client of a real endpoint, caller-selected raw
  HTTP target, or raw upstream SSE remains in the integrated path.
- Real URL/key sentinels are absent from provider control state. Hidden-answer/private-test
  sentinels are absent from case descriptors, primary-model calls, diagnostics, request artifacts,
  and public artifacts; declared restricted judge/verifier bodies are route-confined and unlogged.

### 15.2 Protocol and lifecycle

- Malformed, oversized, unknown, duplicate, late, out-of-order, and missing messages fail closed.
- Worker self-reported provenance cannot substitute for factory-attested executable/source/lock
  identity; schema fingerprints and launch evidence match exactly.
- All queues are bounded; adversarial call volume proves fairness and bounded memory.
- Provider-requested cancellation of one queued or streaming operation produces exactly one Rust
  terminal and leaves no transport attempt live.
- Cancellation while planned, queued, dispatched, streaming, solving, verifying, and finalizing
  leaves no task, operation, process, sandbox, or artifact writer alive.
- Artifact tests retain hostile descendant writers and race path replacement; sealing starts only
  after enforced quiescence and succeeds only through FD-relative no-follow traversal.
- Worker crash and forced shutdown are infrastructure failures, never scores.

### 15.3 Standalone provider compatibility

- NeMo Evaluator `run_evaluation`/CLI through `LocalEvaluationHost` preserves frozen selection, shuffle, shard,
  repeat, seed-once behavior, per-case result, aggregate, and bundle semantics.
- OpenBench `bench eval` retains its standalone task, solver, scorer, reducer, and EvalLog
  semantics; programmatic recording-host and pipe-host runs normalize to the same result.
- Recording-host tests cover every supported operation and purpose.
- Core import guards prevent network/provider implementations from re-entering session logic.
- The integrated OpenBench path uses public Inspect `ModelAPI`, `sandboxenv`, and `eval_async`
  seams and contains no private registry, client-method, display, or recorder mutation.

### 15.4 AIPerf traffic truth

- Primary, environment, verifier, judge, simulator, reward-model, embedding, tool, and other
  supported calls each reconcile exactly once in the Rust ledger.
- Logical calls, transport attempts, retries, cache/replay status, route, tokens, and terminal
  status reconcile under success and failure.
- Clock-driven retry tests cover backoff cancellation, unique attempt lineage, proxy deduplication,
  and fail-closed retry after partial output.
- Rust parses real SSE. A streaming-capable provider receives typed deltas and one terminal;
  Inspect's terminal `ModelAPI.generate` receives one normalized terminal result only.
- Two independent endpoints prove service-based routing without URL/key leakage.

### 15.5 Semantic parity

For every migrated provider:

- exact package/source/dependency/dataset/image identities;
- exact selected cases and order;
- static scheduled modes preserve template order while assigning unique phase/issue/cycle
  occurrences; finite host batches remain single-instantiation;
- byte-equivalent effective prompts, messages, tools, images, and generation controls;
- exact per-case scores/status, primary score, denominator policy, aggregates, category weighting,
  pass@k, confidence intervals, and exclusions;
- valid zero, partial/empty graded failure, infrastructure, and cancellation remain distinct;
- exact required trajectories/provider artifacts after declared nondeterministic normalization;
- public result/artifact projections pass registered schemas, while arbitrary provider strings,
  expected answers, and full provider bundles remain restricted;
- evaluator standalone and AIPerf pipe- or proxy-host frozen-response runs produce the same canonical
  provider bundle/result digest after declared normalization. For OpenBench this includes typed
  score trees, epoch reducers, named metrics, and the restricted normalized EvalLog result.

Aggregate-score equality alone is not parity.

### 15.6 Failure matrix

Tests cover at least:

- 429, 5xx, timeout, DNS/connect/TLS failure, disconnect, malformed SSE, empty/partial response,
  retry exhaustion, and cancellation;
- evaluator crash, invalid JSONL, scorer/judge exception, environment/tool failure, missing asset,
  mutable identity, digest mismatch, sandbox cleanup failure, artifact traversal/symlink, and
  shutdown timeout;
- Inspect installed-entry-point load attempts, missing model-call context, per-sample failure, and
  batch abort; every frozen sample/epoch has one terminal or the run fails infrastructure;
- cache/resume disabled in measured mode and explicitly labeled replay when enabled;
- unsupported browser, opaque/internal batch, HTTP tool/MCP, sandbox, operation, or stream
  capability rejected during preparation; declared `host_batch` execution proves exact
  per-case outcomes, effect accounting, and whole-unit cancellation.

### 15.7 Product reachability and deletion

- Config v2 → Python orchestrator → runner protocol v2 → evaluation provider → Rust HTTP → native
  report has a real subprocess proof.
- Capabilities list only executable provider/backend/workload/operation combinations.
- A provider coverage manifest maps every existing AIPerf benchmark/provider to an exact
  provider-specific parity proof or keeps the old implementation. Similar names or aggregate
  scores are insufficient.
- Report consumers have migrated or receive a versioned deterministic compatibility projection.
- Only then delete the corresponding static or stateful implementation. The final old-path
  deletion occurs only when the entire matrix is green.

---

## 16. Explicit non-goals

- Porting benchmark prompts, hidden tests, agents, or scorers to Rust.
- Allowing evaluator plugins to make arbitrary network calls through a generic Rust proxy.
- Supporting BrowserGym, remote Gym/NAT, HTTP MCP, or opaque batch containers before compliant
  Rust host capabilities exist.
- Making semantic accuracy available in the Dynamo timing simulator.
- Combining mutually incompatible evaluator dependency universes into one environment.
- Nesting OpenBench under NeMo Evaluator, NeMo Evaluator under Inspect, or translating one
  provider's native score algebra into the other's.
- Preserving the static/agentic split as permanent public architecture.
- Depending on upstream acceptance before our implementation can work.

---

## 17. Open implementation details that do not reopen the decision

The following are resolved during increment design/review without weakening the boundary:

- exact Linux syscall-isolation mechanism and portable fail-closed provider availability;
- whether stream event transport remains JSONL or uses a second inherited framed pipe after the
  JSONL conformance proof;
- the final crate/module home for host executor registries, subject to cycle-free dependency
  direction;
- exact compatibility-release duration for legacy report/workload aliases;
- which simple static provider/task is the first real parity canary.

Any proposed answer that gives Python a real endpoint, upstream credential, unrestricted socket,
raw upstream SSE, caller-selected forward URL, or generic network escape hatch is outside this RFC
and requires a new architectural decision. The only exception is the scoped local Rust proxy
defined above.

---

## 18. Adversarial review and default-refute adjudication

Two independent default-refute adjudicators checked every adversarial finding against source and
this RFC. `confirmed` means both confirmed it; `refuted split` means only one confirmed it;
`refuted unanimous` means both refuted it. Only confirmed findings changed the design. Duplicate
corrections are deliberately merged.

| ID | Outcome | Resulting correction or no-change ruling |
|---|---|---|
| NEL-R1 | confirmed | Added provider templates plus Rust-scheduled occurrence identity and fail-closed phase compatibility. |
| NEL-R2 | confirmed | Stock factories own and attest launch identity; merged with SEC-R11. |
| NEL-R3 | confirmed | Added factory-owned fingerprinted schemas and side-effect-free Rust validation. |
| NEL-R4 | confirmed | Added provider-originated per-operation cancellation and proxy-disconnect lowering. |
| NEL-R5 | refuted unanimous | No change: a unit may retain NEL's seed-once source-problem repeat lifecycle. |
| NEL-R6 | refuted split | No change: Rust transport terminals and provider semantic outcomes remain separate axes. |
| NEL-R7 | confirmed | Added enforced process-tree/writer quiescence before sealing; merged with SEC-R7. |
| NEL-R8 | confirmed | Provider data defaults restricted and public projections are factory-schema-owned; merged with SEC-R5. |
| NEL-R9 | confirmed | Added Rust retry-policy/attempt-executor seams, Clock backoff, lineage, and proxy deduplication. |
| INS-R1 | confirmed | The direct proxy decision already distinguished inert semantic URL content from connection authority/dereference. |
| INS-R2 | confirmed | The direct proxy decision already added restricted, unlogged judge/verifier payloads confined to auxiliary routes. |
| INS-R3 | confirmed | Added fail-closed Inspect entry-point policy and frozen explicit registration. |
| INS-R4 | confirmed | Added public task-local model-call context with sample/epoch/attempt/call identity. |
| INS-R5 | refuted split | No change beyond the existing immutable asset resolver and capability fail-closed rules. |
| INS-R6 | confirmed | Pinned Inspect's continue-on-sample-failure profile and exhaustive terminal join. |
| INS-R7 | confirmed | Removed the proposed jitter patch; use public `max_sandboxes=None` and default concurrency `None`. |
| INS-R8 | refuted unanimous | No change to the concrete frozen Task-instance design. |
| SEC-R1 | confirmed | Expanded network isolation into mandatory environment/filesystem/process/resource isolation. |
| SEC-R2 | refuted split | No change: destination-level isolation remains an outcome contract behind platform traits. |
| SEC-R3 | confirmed | Moved protocol traffic to dedicated child-safe control descriptors. |
| SEC-R4 | refuted split | No further adjudicated change; the explicit user decision already added restricted judge payloads through Rust. |
| SEC-R5 | confirmed | Merged with NEL-R8's restricted-default, factory-schema-owned projection rule. |
| SEC-R6 | refuted split | No change to the report's current restricted-artifact digest/manifest design. |
| SEC-R7 | confirmed | Merged with NEL-R7 and added FD-relative no-follow sealing after quiescence. |
| SEC-R8 | refuted unanimous | No change to the two-axis transport-terminal/provider-semantic-outcome model. |
| SEC-R9 | refuted unanimous | No change: reducer membership and epoch semantics remain provider-owned. |
| SEC-R10 | refuted unanimous | No change to the existing typed capability, grant, schema, and sandbox model. |
| SEC-R11 | confirmed | Merged with NEL-R2's factory-owned launch attestation. |
| SEC-R12 | confirmed | Added versioned canonical JSON rules and distinct raw-artifact versus normalized-result digest domains. |

---

## Addendum — 2026-07-12: neutral host foundation and bounded stock canaries built

This addendum supersedes the document header's blanket **not implemented** status and the open
question in section 17 about the first parity canary. The ownership decision, fail-closed security
boundary, staged-migration policy, and acceptance/deletion gates remain authoritative. The neutral
provider/host foundation and two exact GSM8K canaries are built; broad NeMo Evaluator or
OpenBench/Inspect task support and replacement of the legacy static/stateful providers are not.

### Built neutral protocol and runtime

- `crates/aiperf-accuracy/src/{provider,provider_protocol,supervisor,lifecycle}.rs` implements the
  object-safe `EvaluationProvider` / `EvaluationProviderFactory` / launcher seams, deterministic
  availability-filtered registry, strict evaluator-worker protocol v2 DTOs, bounded correlated
  JSONL control over dedicated inherited descriptors 3/4, factory-owned side-effect-free authored
  validators, pre-exec launch-closure attestation, negotiated/final identity checks, cancellation,
  and the complete lifecycle through deferred report commit. Ordinary stdout is not a protocol
  channel, and provider stderr bytes remain restricted.
- `crates/aiperf/src/evaluation/{host,arbiter,ledger,retry,inference,workload}.rs` implements open
  typed host-executor registries, logical Rust-owned routes, bounded fair unit/operation admission,
  exact logical-operation and transport-attempt accounting, Clock-driven retry/backoff,
  cancellation, prepared-endpoint inference lowering, provider event draining, and the generic
  evaluation workload. Provider code still owns prompts, solvers, scorers, semantic outcomes, and
  canonical aggregates.
- `crates/aiperf-runner/src/evaluation_execution.rs` supplies the protocol-v2
  `online_http + evaluation` adapter, immutable GSM8K asset resolver, prepared HTTP route
  composition, native metrics join, and runner report commit. Python authors only a registered
  provider/distribution, opaque provider configuration, logical routes, and bounded concurrency;
  it supplies no executable, import target, environment, endpoint URL, or credential.

The legacy `AccuracyEvaluator`, `AgenticEvaluator`, `static_accuracy`, and `agentic` surfaces remain
exported and registered during migration. The new provider protocol does not silently reinterpret
either legacy workload.

### Built launch isolation and process-tree proof

`crates/aiperf-accuracy/src/isolation.rs` and
`crates/aiperf-runner/src/stock_evaluation.rs` implement the stock
`linux-bubblewrap-rootfs-process-tree-v4` profile. Rust independently resolves and hashes the exact
registered CPython, provider package/source-overlay/dependency-lock, system-library, task-manifest,
and asset closure; materializes a fresh single-link worker root; mounts it read-only at `/`; and
adds only private writable staging plus the optional per-run Unix proxy socket. Bubblewrap runs the
worker as uid/gid 65534 with `--unshare-all`, all capabilities dropped, a cleared allowlisted
environment, private `/proc`, resource ceilings, descriptor confinement, `no_new_privs`, and no
external network. The supervisor pins the worker and every observed descendant by PID/start time,
captures each private PID-namespace init, and must prove the complete tree exited before sealing.

The profile is currently Linux x86_64 and fail-closed. An unavailable or digest-drifted runtime,
provider root, system closure, Bubblewrap binary, source overlay, lock, manifest, or asset makes
that distribution unavailable rather than weakening isolation.

### Built scoped compatibility proxy

`crates/aiperf/src/evaluation/proxy.rs` implements a per-run Unix-domain HTTP/SSE proxy. It accepts
only registered local dialect paths and schema-validated operation payloads; it has no caller-
selected upstream URL, forwarding method/header surface, real credential, or raw upstream SSE.
Rust mints the grant, narrows it after provider planning to the exact cases/routes/purposes/
operations and byte/concurrency/lifetime budgets, authenticates the Unix peer against the attested
worker process subtree with kernel credentials, and routes accepted work into the same fair queue,
host executor, transport-attempt ledger, cancellation, usage, and report path as pipe operations.
Proxy timing and bounded shutdown use the run's injected `Clock`.

The stock NeMo canary is pipe-only. The stock OpenBench canary uses only the registered
`openai_chat_completions` local dialect for its candidate `model.generate` route. No generic
Internet proxy is present.

### Built artifact and report foundation

`crates/aiperf-accuracy/src/artifacts.rs`,
`crates/aiperf/src/evaluation/{workload,report}.rs`, and the generic evaluation types in
`crates/aiperf-metrics/src/report.rs` enforce the terminal order
drain/finalize → worker process-tree quiescence → proxy shutdown and socket removal →
FD-relative no-follow artifact verification/promotion → atomic native-v2 report commit.
Symlinks, hard links, devices, traversal, undeclared files, size/digest drift, and mutation during
sealing fail closed. Provider artifacts and the canonical bundle are restricted unless a Rust
factory owns an exact path/media/schema public-projection rule.

The native-v2 `evaluation` block publishes factory-projected safe config, opaque case/artifact
identities, reviewed score/numeric/aggregate projections, separate completed/infrastructure/
cancelled counts, Rust-owned route and attempt summaries, exact identity/source-overlay/lock
digests, and Rust-computed artifact/result digests. Rust validates and publishes the provider's
aggregate value and counts under a factory-bound definition; it does not recompute the provider's
scorer or reducer. Unregistered provider metadata and scores fall back to opaque/restricted rather
than becoming public strings.

### Exact stock GSM8K canaries

Both stock distributions bind the same five-record immutable asset
`openai_gsm8k_main_test_canary`, sourced from
`openai/gsm8k@740312add88f781978c0658806c59bc2815b9866:main:test:first5` with content SHA-256
`fc9b5c03206d193c0013baf2d6344a133fe0096a2b47cd1eafdcee297dfd398a`. Their only declared host
operation is chat-shaped `model.generate`.

| Provider | Exact built surface |
|---|---|
| NeMo Evaluator | `nvidia_nemo_evaluator_0_4_locked`: NeMo Evaluator 0.4.0 at `a668af906b46c802984f2d471f15ca83b763092d` plus two pinned generic host/session overlays. Only `environment=gsm8k`, `solver=chat`, finite case granularity, selection seed 0, and limit 1–5 are accepted. Model traffic uses the typed pipe host. |
| OpenBench / Inspect | `groq_openbench_0_5_3_inspect_0_3_141_locked`: OpenBench 0.5.3 at `3f190a835f7fee34ccd96e17242a36a29e0620a6` plus Inspect AI 0.3.141 at `bb78d82dde311b68dbfd0b49f3186b9fc13a1465`, with pinned explicit-runtime, public model-call-context, cache-veto, and entry-point-deny overlays. Only `task=gsm8k`, finite host-batch granularity, limit 1–5, and epochs 1–8 are accepted. The provider uses Inspect's public task/model APIs through the scoped local proxy; cache, provider retry/batch, arbitrary entry points, task arguments, and sandboxes are disabled. |

Each provider retains its native GSM8K prompt preparation, score name, reducer definition, and
restricted canonical bundle/EvalLog semantics. The only reviewed public case-score shape is the
exact binary object `{"value": 0}` or `{"value": 1}` under
`gsm8k_binary_score_v1`; NeMo's `reward/mean` and Inspect's
`grade_school_math_scorer` identity/epoch semantics remain distinct provider-owned definitions.
The coverage manifest deliberately records a one-executed-case parity proof for each provider,
not full GSM8K migration. Provider-level parity tests cover standalone preservation, deterministic
NeMo selection/seed behavior, public Inspect execution/fail-closed policies, and absence of
private-Inspect mutation or provider-owned HTTP. The ignored real-root runner subprocess proof in
`crates/aiperf-runner/tests/evaluation_process.rs` exercises both distributions through Rust-owned
HTTP/SSE, verifies public redaction and artifact cleanup, and checks the exact capability surface.

### Product registration is deployment-root conditioned

The runner does not advertise evaluation merely because the Rust factories are linked.
`stock_evaluation_composition()` freezes each distribution only after its complete launch closure
and isolation implementation pass availability checks. `registry.rs` registers the
`evaluation` workload and `online_http + evaluation` pair only when at least one stock
distribution is executable, and capability combinations contain only the distributions and
`model.generate` adapter that passed those checks. With no deployment-owned roots, the evaluation
provider inventory and supported combinations are empty and the pair is absent.

`src/aiperf/orchestrator/runner_installation.py` binds discovery to the selected runner deployment:
an installed companion consumes only its own wheel-`RECORD`-verified evaluator-root registry,
while an explicitly selected/PATH runner consumes only a generated adjacent sidecar. Missing,
incomplete, symlinked, extra, or digest-drifted roots yield no evaluator roots; ambient
`AIPERF_EVALUATOR_PROVIDER_ROOTS` cannot broaden product selection. The selected roots are passed
only to capability negotiation and the fresh runner child.

### Migration and deletion gates remain open

`src/aiperf/accuracy/evaluation/manifests/provider_coverage.json` is authoritative for migration
scope. It retains every complete static benchmark—including the full GSM8K split and configuration
matrix—on its named legacy benchmark/grader path, and retains Harbor 0.18, AgentLab/BrowserGym,
and MCPMark on their existing stateful paths. The two five-record stock canaries do not establish
parity for any other NeMo Evaluator environment, OpenBench task/group, full GSM8K execution,
`rust_occurrences`, auxiliary model role, sandbox/process/resource/MCP/browser effect, gRPC or
offline evaluation backend, or legacy report consumer.

Those capabilities remain unsupported until an exact frozen manifest, executable Rust host
adapter, isolation proof, provider parity evidence, product subprocess proof, and the original
section 15 deletion gates all pass. The long-term full-replacement decision therefore stands, but
no legacy provider or workload may be deleted on the evidence of these canaries alone.
