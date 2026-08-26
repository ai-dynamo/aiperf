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
| S04 | HTTP-owned shared transport configuration | Confirmed/narrowed; spec ready | Sol implementation plan |
| S05 | HTTP hot path allocates/reparses unnecessarily | Confirmed/narrowed; spec ready | Sol implementation plan |
| S06 | GPU worker mutex spans IPC awaits | Obsolete | Record retraction |
| S07 | Unbounded graph/media channels | Confirmed/narrowed; spec ready | Sol implementation plan |
| S08 | Cellular test helper has a vacuous missing-log path | Confirmed; spec ready | Sol implementation plan |
| S09 | Stale `dead_code` suppressions on live fields | Already fixed | Record retraction |
| S10 | Disagg `TraceCollector` API suppressions lack coherent disposition | Confirmed/narrowed; spec ready | Sol implementation plan |
| S11 | `Inputs` is an unguarded god struct | Confirmed/narrowed; spec ready | Sol implementation plan |
| S12 | Cellular controller feature gates are scattered | Confirmed; spec ready | Sol implementation plan |
| S13 | Graph benchmark bypasses the transport seam | Confirmed/narrowed; spec ready | Sol implementation plan |
| S14 | DAG validation test does not prove rejection | Confirmed; spec ready | Sol implementation plan |
| S15 | `ScheduledPhasePlan` booleans lack semantic prefixes | Confirmed; spec ready | Sol implementation plan |
| E01 | Velo route registration does not bind signed peer identity | Confirmed; rationale spec ready | Sol implementation plan |
| E02 | Externally-driven evaluation skips preflight | Confirmed; spec ready | Sol implementation plan |
| E03 | NativeGraph execution ignores agent timeout | Confirmed; spec ready | Sol implementation plan |
| E04 | Abort still starts ready tool nodes | Confirmed; spec ready | Sol implementation plan |
| E05 | Closed-worker credit return can hang drain | Confirmed; spec ready | Sol implementation plan |
| E06 | Sidecar setup/finish failures leak siblings | Confirmed/narrowed; spec ready | Sol implementation plan |
| E07 | Global-push cancellation is misclassified as transport failure | Confirmed; spec ready | Sol implementation plan |
| E08 | Declared terminal outputs cannot execute | Confirmed; spec ready | Sol implementation plan |
| E09 | Multi-dataset requests silently retain only the first | Confirmed; spec ready | Sol implementation plan |
| E10 | Docker verifier orchestration is duplicated and diverges | Confirmed; spec ready | Sol implementation plan |
| E11 | CLI Dynosim runs skip process defaults | Confirmed; spec ready | Sol implementation plan |
| C01 | Phaser Velo subscription accepts forged/replayed event | Confirmed; rationale candidate | Benchmark-proportional disposition |
| C02 | Dataset Velo subscription accepts forged event or oversized payload | Confirmed/narrowed; spec needed | Sol implementation plan |
| C03 | Duplicate cellular partition can satisfy completion barrier | Confirmed; spec needed | Sol implementation plan |
| C04 | Eventstream prelude validation happens after accumulation | Confirmed; spec needed | Sol implementation plan |
| C05 | Partial final HTTP frame lacks terminal decode error | Confirmed; spec needed | Sol implementation plan |
| C06 | Authored HTTP body cap is not propagated to client config | Confirmed; spec needed | Sol implementation plan |
| C07 | H2 prior knowledge is not usable over UDS | Confirmed; spec needed | Sol implementation plan |
| C08 | Graph successor waits for parent completion instead of first token | Confirmed; spec needed | Sol implementation plan |
| C09 | Graph firing delay ignores cancellation | Confirmed; spec needed | Sol implementation plan |
| C10 | Graph cancellation still permits tool successor | Confirmed; spec needed | Sol implementation plan |
| C11 | Graph-runtime panic leaks idle accounting | Confirmed; spec needed | Sol implementation plan |
| C14a | YAML accepts boolean `artifacts.records` outside the contract | Fix exists on divergent branch | Rebase/cherry-pick with TDD receipt |
| C14b | YAML accepts unsupported schema versions | Confirmed; spec needed | Sol implementation plan |
| C18 | MLflow/W&B mishandle mixed labeled and unlabeled report series | Confirmed; spec needed | Sol implementation plan |
| C19 | Parquet histogram schema drifts across samples | Confirmed; spec needed | Sol implementation plan |
| C20 | Dataset-analysis writer loses flush failures | Fix exists on divergent branch | Rebase/cherry-pick with TDD receipt |
| C24 | Slot release during a decrease can over-admit | Confirmed; spec needed | Sol implementation plan |
| C25 | Global request-rate prefill release does not wake issuer | Confirmed; spec needed | Sol implementation plan |
| R01 | YAML/CLI config loses `--image-batch-size` | Already fixed | Preserve existing regression test |
| B01 | AgentX integration fixtures omit required cache-bust option | Complete — `91b65b2044` | Independent Graham approved; compile regression green |

## Progress log

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
