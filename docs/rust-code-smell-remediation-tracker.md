# Rust Code-Smell Remediation Tracker

This document is the authoritative progress ledger for the requested
remediation campaign. It records each concrete claim found in the most recent
Rust-focused smell/review artifacts, its analysis, the specification and plan
receipts, implementation evidence, RED-to-GREEN tests, and Graham review.

## Source inventory

| Source | Claims included | Selection rationale |
| --- | --- | --- |
| `.superpowers/sdd/code-smell-review/findings.md` | Claims 1, 3-15 | Explicit Rust code-smell claim inventory. Claim 2 is retracted by its source and is not actionable. |
| `.superpowers/sdd/eval-altitude-review/findings.md` | Findings 1-11 | Recent Rust correctness, liveness, and structural findings. |
| `.superpowers/sdd/2026-08-25-rust-smell-remediation/verification-matrix.md` | C01-C25 | Active prior Rust smell-remediation inventory; rows remain in scope until their own lifecycle receipts are verified. |
| `artifacts/code-review.md` | P1 | Most recent in-repository Rust behavior audit. |

## Required lifecycle

Every row remains open until all of the following receipts exist:

1. Current-code analysis confirming, narrowing, or retracting the claim.
2. A precise, testable specification.
3. A plan authored by a `gpt-5.6-sol` planning agent from that specification.
4. SDD implementation with test-first RED and verified GREEN evidence.
5. Focused and relevant full test execution using `CARGO_TARGET_DIR` under
   `/mnt/4tb`.
6. A completed Graham code review with no unresolved blocking finding.
7. A commit containing the finished remediation.

## Benchmark proportionality ruling

This is a benchmarking tool, not a general-purpose security product. Each
security-flavoured finding is evaluated for its effect on benchmark correctness,
measurement fidelity, resource bounds, or local operator diagnosability. If a
finding only improves hardening outside those goals, its specification may close
with a small, explicit in-code or adjacent documentation rationale rather than
new authentication, isolation, or operational machinery. The rationale must
state the trust assumption, why the behavior is acceptable for this tool, and
which measurement-relevant invariant remains protected. It still receives a
Graham review and a commit.

## Inventory

| ID | Source claim | Status | Next receipt |
| --- | --- | --- | --- |
| S01 | Socket-option failures are silently discarded | Complete — `ec97423efa` | Graham approved; focused tests green |
| S03 | Transport-neutral dispatch hardwires `NativeMetricsObserver` | Complete — `fae122a100` | Independent Graham approved; RED→GREEN evidence recorded |
| S04 | HTTP-owned shared transport configuration | Complete | Integrated `164955a757`, `9042df24d3`; independent Graham r2 PASS |
| S05 | HTTP hot path allocates/reparses unnecessarily | Complete — `d226475e39`, `aaa022f4c6`, `ad4eeb4066` | Independent Graham r3 approved; prepared/direct and bounded-dispatch RED→GREEN evidence recorded |
| S06 | GPU worker mutex spans IPC awaits | Obsolete | Record retraction |
| S07 | Unbounded graph/media channels | Complete — `9c32b959e3`, `d5136f005e`, `c6e681abd7`, `66eaaf89b5` | Independent Graham r3 approved; RED→GREEN lifecycle and finalization evidence recorded |
| S08 | Cellular test helper has a vacuous missing-log path | Complete — `860d93510b` | Independent Graham approved; RED→GREEN e2e evidence recorded |
| S09 | Stale `dead_code` suppressions on live fields | Already fixed | Record retraction |
| S10 | Disagg `TraceCollector` API suppressions lack coherent disposition | Complete on integration branch — `bacf85c014` | Independent Graham approved; await shared-file integration |
| S11 | `Inputs` is an unguarded god struct | Complete — `de8b24adb6`, `37a78fd0a2`, `2fc4d17c6e` | Independent Graham r2 approved; direct grouped-construction and flat-wire evidence recorded |
| S12 | Cellular controller feature gates are scattered | Complete — `d1948cd50d` | Independent Graham approved; feature-on controller behavior and parent-gate boundary evidence recorded |
| S13 | Graph benchmark bypasses the transport seam | Complete — `0e2e3f7aa1` | Independent Graham approved; default API boundary, opt-in transport feature, and DynoSim report evidence recorded |
| S14 | DAG validation test does not prove rejection | Confirmed; spec ready | Sol implementation plan |
| S15 | `ScheduledPhasePlan` booleans lack semantic prefixes | Complete — `13fdae4430` | Independent Graham approved; RED→GREEN policy coverage recorded |
| E01 | Velo route registration does not bind signed peer identity | Complete | Integrated `ef7b966238`, `56e35146bb`; independent Graham r2 PASS |
| E02 | Externally-driven evaluation skips preflight | Complete — `b488309a2c` | Independent Graham approved; RED→GREEN pre-side-effect Docker-capability coverage recorded |
| E03 | NativeGraph execution ignores agent timeout | Confirmed; spec ready | Sol implementation plan |
| E04 | Abort still starts ready tool nodes | Already fixed — `dfa0043510`, `af44cb48ef` | Duplicate of completed C10; independent Graham approval and behavioral RED→GREEN evidence are recorded in `6bc3f9f988` |
| E05 | Closed-worker credit return can hang drain | Confirmed; spec ready | Sol implementation plan |
| E06 | Sidecar setup/finish failures leak siblings | Complete | Integrated `bb5e3fe5ff`, `bb23c66b39`; independent Graham PASS |
| E07 | Global-push cancellation is misclassified as transport failure | Complete | Integrated `82bf74990c`; independent Graham PASS |
| E08 | Declared terminal outputs cannot execute | Complete | Integrated `dcf1454576`, `caa260e9dc`, `405e54c2b1`; independent Graham PASS |
| E09 | Multi-dataset requests silently retain only the first | Complete | Integrated `e8a787f47e`; independent Graham PASS |
| E10 | Docker verifier orchestration is duplicated and diverges | Complete | Integrated `34ac902468`, `101bf1a3d2`; independent Graham PASS |
| E11 | CLI Dynosim runs skip process defaults | Complete | Integrated `50973a484a`; independent Graham PASS |
| C01 | Phaser Velo subscription accepts forged/replayed event | Complete — `cf09af5034` | Independent Graham approved; explicit benchmark trust-boundary contract |
| C02 | Dataset Velo subscription accepts forged event or oversized payload | Complete — `c823d9c53e` through `6bb0066d47` | Independent Graham approved; RED→GREEN Velo fan-out bound, replay, and progress coverage recorded |
| C03 | Duplicate cellular partition can satisfy completion barrier | Complete — `1053327b41` | Independent Graham approved; RED→GREEN identity coverage recorded |
| C04 | Eventstream prelude validation happens after accumulation | Complete — `4067dfc13b`, `7b0d6fce79` | Independent Graham approved; RED→GREEN parser and mock-caller coverage recorded |
| C05 | Partial final HTTP frame lacks terminal decode error | Complete — `d2c92bae08` | Independent Graham approved; RED→GREEN EOF coverage recorded |
| C06 | Authored HTTP body cap is not propagated to client config | Complete — `0a9ba67232` | Independent Graham approved; clean-head behavioral RED→GREEN coverage recorded |
| C07 | H2 prior knowledge is not usable over UDS | Complete — `0ed7e8980d` | Independent Graham approved; UDS H1/H2 RED→GREEN coverage recorded |
| C08 | Graph successor waits for parent completion instead of first token | Complete — `f363004e9d`, `df6e18adb7` | Independent Graham approved; focused RED→GREEN and 320 graph tests green |
| C09 | Graph firing delay ignores cancellation | Complete — `77e606b065`, `676025af64` | Independent Graham approved after deterministic post-token delay coverage; 321 graph tests green |
| C10 | Graph cancellation still permits tool successor | Complete — `dfa0043510`, `af44cb48ef` | Independent Graham approved after deterministic pre-dispatch and in-flight cancellation coverage; 323 graph tests green |
| C11 | Graph-runtime panic leaks idle accounting | Complete — `adea330ae8` | Independent Graham approved; panic RED→GREEN and 8 graph-runtime tests green |
| C14a | YAML accepts boolean `artifacts.records` outside the contract | Complete — `1999e28b54`, `55ce407c1d` | Independent Graham approved; isolated behavioral RED→GREEN and 42 YAML tests green |
| C14b | YAML accepts unsupported schema versions | Complete — `bec9ace814`, `3ffb9be6d7`, `711d4c2e76`, `556611fc3d` | Independent Graham approved after null/non-string hardening; 44 YAML tests green |
| C18 | MLflow/W&B mishandle mixed labeled and unlabeled report series | Complete — `b98d0b4b8f` | Independent Graham approved; four-case RED→GREEN and exporter suites green |
| C19 | Parquet histogram schema drifts across samples | Complete — `d4db61806e` | Independent Graham approved; schema-expansion RED→GREEN and 9 Parquet tests green |
| C20 | Dataset-analysis writer loses flush failures | Complete — `ed39aac8a4` | Independent Graham approved; dataset-analysis JSON tests green |
| C24 | Slot release during a decrease can over-admit | Complete | Integrated `47795ff8c0` through `96787c5a15`; independent Graham PASS |
| C25 | Global request-rate prefill release does not wake issuer | Complete — `bdeb53dda4` | Independent Graham approved; focused wake RED→GREEN plus request-rate and slots suites green |
| R01 | YAML/CLI config loses `--image-batch-size` | Already fixed | Preserve existing regression test |
| B01 | AgentX integration fixtures omit required cache-bust option | Complete — `91b65b2044` | Independent Graham approved; compile regression green |

## Progress log

- 2026-08-26: Reconciled E04 to completed C10. The E04 requirement is the
  same pre-abort, in-flight-abort, and no-post-abort-publication behavior
  implemented by `dfa0043510` and `af44cb48ef`; C10 already has independent
  Graham approval and deterministic execution-level coverage. No duplicate
  production change is required.

- 2026-08-26: Completed S13. Raw HTTP graph transport benchmarking is now an
  explicit non-default feature, while the transport-neutral graph report type
  is owned by `graph::report` and consumed by DynoSim. Default API absence,
  opt-in compilation, and focused DynoSim regressions were RED→GREEN; Graham
  review approved the final series.

- 2026-08-26: Completed S12. The parent cellular-controller module gate is now
  the sole feature boundary; 47 redundant inner `cellular` attributes were
  removed while Unix/test gates remain. A fresh feature-on controller test and
  independent Graham review passed; feature-off errors were independently
  localized to unchanged non-controller baseline modules.

- 2026-08-26: Completed E02. Externally driven execution now checks Docker
  capability before consuming the prepared Driver capability or triggering any
  Driver/container/build side effect. The ordering regression was RED then
  GREEN, the Docker runtime suite passed 79 tests, and independent Graham
  review approved the change.

- 2026-08-26: Completed S05. Prepared HTTP requests now retain their parsed
  URL and typed static headers through ordinary and bounded SSE dispatch;
  artifact header maps are only materialized for raw capture, while direct
  callers retain their `BTreeMap` facade. Behavioral RED→GREEN wire tests and
  independent Graham r3 approved the complete series.

- 2026-08-26: Completed S11. Profile input ownership now resides in seven
  typed groups constructed directly by both CLI and YAML paths; the flat
  externally serialized execute wire remains compatible, without a large
  intermediate DTO or JSON conversion bridge. Five authoring regressions and
  independent Graham r2 approved the final series.

- 2026-08-26: Completed S07. Graph and media event delivery now uses bounded,
  nonblocking queues: graph saturation deterministically fails and releases the
  phase lifecycle, while media overflow is latched and fails sidecar
  finalization. The saved RED timeout and both behavioral GREEN regressions
  passed; independent Graham r3 approved the complete series.

- 2026-08-26: Completed C11. Graph-runtime in-flight accounting is now an RAII
  responsibility, so a spawned task panic decrements and notifies idle waiters
  during unwind. Independent Graham review approved the panic regression.

- 2026-08-26: Completed C25. A global prefill-capacity release or limit growth
  now wakes a capacity-saturated issuer through a no-missed-wake protocol,
  without changing session-capacity behavior. Independent Graham review
  approved the regression and targeted suites.

- 2026-08-26: Completed C14b. Every authored schema version is now validated
  before expansion and after expanded/sweep normalization: only the exact string
  `"2.0"` is accepted; null and non-string values receive the same explicit
  diagnostic. Independent Graham approval followed the P1 hardening.

- 2026-08-26: Completed C10. Tool successors now stop both before dispatch and
  while dispatch is in flight when their trace is cancelled. The final regression
  deterministically parks a tool dispatch before cancellation; independent
  Graham review approved the repair.

- 2026-08-26: Completed C19. Parquet histogram export now retains the
  deterministic union of observed bounds and remaps prior samples with zeroes
  for newly introduced buckets. Independent Graham review passed nine Parquet
  exporter tests.

- 2026-08-26: Completed C09. Cancellation now races both the parent first-token
  wait and the clock-injected firing delay; the regression waits until the child
  is demonstrably parked at the post-token simulated deadline before canceling.
  Independent Graham approval followed the coverage repair.

- 2026-08-26: Completed C18. MLflow and W&B now share the same series
  selection rule: use the unique unlabeled aggregate when labels are mixed,
  and avoid emitting misleading labeled-only metrics. Independent Graham review
  passed the MLflow (10) and W&B (12) suites.

- 2026-08-26: Completed C14a. `artifacts.records` now accepts only a format
  list or `false`; `true` is rejected rather than silently disabling records.
  The pre-fix witness failed behaviorally, the fixed worktree passed 42 YAML
  tests, and the independent Graham review found no issues.

- 2026-08-26: Completed C08. First-token successor readiness now includes
  static timing gates and resilient terminal paths retain completion fallback;
  the independent Graham rerun passed four focused tests and the 320-test graph
  library suite.
- 2026-08-26: Completed C20. Dataset-analysis JSON persistence now propagates
  buffered-writer flush failures instead of reporting an artifact that may not
  have reached its sink; independent review passed its focused tests.

- 2026-08-26: Completed E10. Ordinary, multi-step, and NativeGraph Docker
  verification now share the same verifier transaction, including selected
  test-root handling, cleanup, deadline, and joined-error behavior. The
  selected-root behavioral RED passed after repair, the 78-test Harbor suite
  passed (three Docker tests ignored), and the independent Graham re-review
  passed before integration as `34ac902468` and `101bf1a3d2`.

- 2026-08-25: Created the inventory and lifecycle criteria. Existing unrelated
  working-tree changes in `.dockerignore` and `artifacts/code-review.md` were
  preserved.
- 2026-08-25: Added the active prior-remediation matrix to prevent existing
  reported defects from being silently omitted; observed dedicated C14a and C20
  worktrees under `/mnt/4tb` and will reconcile their commits and receipts.
- 2026-08-25: Ruling: use benchmark-proportional security remediation — protect
  correctness, fidelity, bounds, and local diagnostics; close purely
  out-of-scope hardening with an explicit trust-assumption rationale rather than
  unnecessary infrastructure.
- 2026-08-25: Current-code review completed for S01 and S03-S15. S06 was
  superseded by the bounded native vendor-worker architecture and S09 had
  already been repaired; both require only recorded dispositions. The remaining
  findings now have testable remediation specifications awaiting Sol plans.
- 2026-08-25: Current-code review completed for E01-E11 and R01. R01 is
  already covered by a resolved-config regression test. E01 is a real route
  binding hardening gap; its disposition is assessed against the stated local,
  benchmark trust model before implementation. E02-E11 have testable
  remediation specifications awaiting Sol plans.
- 2026-08-25: Active C-series matrix reconciliation found 17 defects still
  present on this branch. C14a and C20 have unverified fixes only on a divergent
  branch and will be reapplied with fresh test evidence. C01 is a
  benchmark-trust-model rationale candidate; C02 remains in scope for its
  resource bound even though its authenticity portion shares that trust model.
- 2026-08-25: S01 completed in `ec97423efa`: required TCP socket options now
  fail connection establishment before handshake, while optional tuning remains
  best-effort. RED evidence, focused GREEN tests, independent Graham PASS, and
  an unsandboxed live-socket integration run are recorded in the SDD task
  artifact.
- 2026-08-25: B01 was discovered while executing S03's required all-target
  regression command. The isolated AgentX fixture repair recorded a genuine
  compile RED (`E0063` at both literals), added the explicit false default,
  passed the exact all-target GREEN command using `/mnt/4tb`, passed an
  independent two-pass Graham review, and was integrated as `91b65b2044`.
- 2026-08-25: S03's scoped implementation is present in externally advanced
  shared commit `fae122a100`. Its original all-target verification is being
  rerun now that B01 has restored integration-target compilation; it remains
  open pending that evidence and independent Graham approval.
- 2026-08-25: S03 completed in `fae122a100`. It has a compile-failure RED,
  a full-path behavior GREEN (1/1), an unblocked all-target compile gate, and
  a focused engine suite GREEN (16/16). An independent Graham review of only
  the six planned source paths passed with no findings. The task commit also
  contains unrelated user-owned changes, which were explicitly excluded from
  the remediation review scope.
- 2026-08-25: The Task 3 S03 record-finalization coverage was green before a
  production edit, proving the Task 2 engine wrapper had already supplied the
  planned behavior. It is therefore a documented test-only plan correction,
  not a manufactured TDD production change: its two success/failure regressions
  and focused 18-test suite are green and independent Graham review passed.
  Reviewed commit `ab2c390e3c` is staged on an integration branch because a
  concurrent user-owned change currently modifies the same shared file.
- 2026-08-25: S08 completed in `860d93510b`: the cellular E2E helper now
  distinguishes absent/unreadable logs from a readable log with no matching
  entries. It has an `E0425` test-first RED, focused and pinned-CLI E2E GREEN
  receipts, and an independent Graham PASS.
- 2026-08-25: S10 completed its isolated remediation as `bacf85c014`: all
  removed APIs were verified producerless outside collector-local tests;
  compatibility fields now serialize as explicit zero/null values and the live
  Dynosim SLA path remains. Exact and 37-test focused GREEN receipts plus
  independent Graham PASS are recorded. It awaits shared-file integration
  behind a concurrent user-owned `collector.rs` edit.
- 2026-08-25: S15 completed in `13fdae4430`: eight phase switches are now
  semantic `ScheduledPhasePolicy` fields with legacy fluent builders preserved.
  The missing-type RED, compiled all-target gate, executed focused policy GREEN,
  and independent Graham PASS are recorded.
- 2026-08-25: Follow-on S15 runtime-effects verification found and repaired a
  real consumer omission in `e004061d87`: local-measurement discard now gates
  native/compatibility metric planes while keeping timing lifecycle records.
  The task has behavioral RED, focused/simulation/online/broad GREEN evidence,
  and an independent Graham PASS.
- 2026-08-25: C01 completed in `cf09af5034` with the approved, test-pinned
  benchmark trust disclosure: raw live controller pushes have no per-push
  authenticity or replay guarantee, while generation ordering remains the
  benchmark-correctness invariant. It adds no runtime security mechanism and
  has RED→GREEN preservation evidence plus independent Graham PASS.
- 2026-08-26: C03 completed and integrated as `1053327b41`: terminal cellular
  receipts are keyed by validated cell identity, reject duplicate/out-of-range
  and mixed-kind receipts, and complete only once every expected cell is
  represented. Its three focused and 45 controller tests are green; independent
  Graham review passed.
- 2026-08-26: C04 completed and integrated as `4067dfc13b` and `7b0d6fce79`:
  EventStream validates prelude fields before retaining a frame body, bounds
  retained frames at the supported protocol maximum, and propagates decode
  failure exactly once. RED→GREEN runtime boundary tests and the mock-server
  caller regression are green; independent Graham review passed.
- 2026-08-26: C05 completed and integrated as `d2c92bae08`: a truncated
  EventStream frame at response EOF produces one terminal SSE decode error and
  then ends, while a clean boundary ends without an error. Focused/module
  GREEN evidence and independent Graham approval are recorded.
- 2026-08-26: C06 completed and integrated as `0a9ba67232` after a clean-head
  TDD redo: an authored positive response-body cap projects unchanged, zero is
  rejected before transport construction, and the existing shared declared and
  chunked enforcement remains covered. The redo has an unknown-field behavioral
  RED, focused GREEN, and independent Graham approval.
- 2026-08-26: C02 completed and integrated as `c823d9c53e` through
  `6bb0066d47`. It applies only to opt-in Velo MessagePack/zstd endpoint-ready
  request fan-out (the distinct HTTP+zstd file/path dataset-source path remains
  unchanged): compressed framing reserves Velo overhead, producer admission
  validates the prospective sealed replay before publication, replay size is
  tracked incrementally, and live delivery reports controller failures or a
  progress-resetting inactivity timeout. The final 13-test scoped suite and
  independent Graham review passed.
- 2026-08-26: C07 completed and integrated as `0ed7e8980d`: the endpoint
  policy preserves explicit H2 prior knowledge over Unix sockets while Auto and
  H1 retain H1 handshakes. Registry projection and live Unix H1/H2 listener
  RED→GREEN tests passed; independent Graham review passed.
- 2026-08-26: E06 completed and integrated as `bb5e3fe5ff` and `bb23c66b39`:
  sidecar setup failure finishes all already-started siblings in reverse order,
  finish failure continues through remaining siblings, and cleanup failures are
  retained as context on the primary error. Its behavioral RED, 7-test focused
  GREEN suite, and independent Graham PASS are recorded.
- 2026-08-26: C24 completed and integrated as `47795ff8c0` through
  `96787c5a15`: a decrease reserves debt before draining, every admission
  rejects reserved debt, and async waiters use a no-missed-wake notification
  protocol when debt clears through release, increase, or drain. Baseline
  over-admission RED evidence, the 16-test slots suite GREEN, and independent
  Graham re-review PASS are recorded.
- 2026-08-26: E07 completed and integrated as `82bf74990c`: a worker-owned
  global-push cancellation settles its issued credit as `Canceled` with a
  typed `dispatch_cancelled` result instead of a transport failure, while a
  real worker transport error remains failed and latches abort-on-failure. Its
  behavioral RED, two focused GREEN tests, post-commit hook audit, and
  independent Graham PASS are recorded.
- 2026-08-26: E09 completed and integrated as `e8a787f47e`: Config-v2 YAML
  accepts exactly one expanded dataset or one shorthand dataset, rejects
  explicit empty/multiple and mixed forms before lowering, and preserves the
  absent-form synthetic default. Its first-entry-loss RED, 47-test YAML GREEN
  suite, and independent Graham PASS are recorded.

- 2026-08-26: E08 completed and integrated as `dcf1454576`, `caa260e9dc`, and
  `405e54c2b1`: executor-owned terminal catalogs now freeze declared staged
  outputs, preserve earlier dynamic handles, and publish the selected final
  handle through the actual `execute_trace` terminal event. A pre-fix terminal
  handle RED, clean focused GREEN, and independent Graham PASS are recorded.

- 2026-08-26: E11 completed and integrated as `50973a484a`: the CLI resolves
  execute input once before cellular selection, Dynosim default selection, and
  execution; the cell path consumes the resolved run as well. The six-test CLI
  and seven-test cellular Dynosim suites are green, with independent Graham
  PASS.

- 2026-08-26: S04 completed and integrated as `164955a757` and `9042df24d3`:
  neutral execution carries only the shared transport policy, while HTTP keeps
  each resolved endpoint's complete client configuration. The initial review
  found graph HTTP and scheduled WebSocket profile-loss regressions; the P1
  repair pins both bindings and the independent Graham r2 review passed.

- 2026-08-26: E01 completed and integrated as `ef7b966238` and `56e35146bb`:
  the authenticated registration seam now records the benchmark's trusted-cell
  route assumption without claiming per-push authenticity, replay protection,
  or transport confidentiality. The review removed a vacuous source-string
  test; real registration/replay and replay/live coverage remain, and the
  independent Graham r2 review passed.
