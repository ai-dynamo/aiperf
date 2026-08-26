# Rust Evaluation and Liveness Remediation Specification

## Scope

This specification covers E02-E11 in the remediation tracker. It protects
benchmark execution correctness, timeout behavior, accounting, and artifact
isolation; it does not widen the benchmark product's security posture.

## E01: Record the trusted benchmark-cell route assumption

Authenticated controller/cell route replacement is a real hardening gap, but
the benchmark's local trusted-controller/cell model does not require a second
per-push authentication protocol. Add a concise rationale adjacent to the
route-admission seam stating that registered benchmark cells are trusted, that
per-push route authenticity/replay protection is deliberately out of scope, and
that this decision must be revisited if the cellular topology admits untrusted
participants. Preserve legitimate replay/live-delivery tests. Do not add
authentication infrastructure or claim transport confidentiality.

## E02: Preflight externally driven Docker runs

Call common Docker preflight before any externally-driven build, provisioning,
or driver side effect. A missing provider capability returns
`UnsupportedEnforcement` and produces no runtime events. Existing external
driver restrictions remain unchanged.

## E03: Enforce the absolute NativeGraph agent timeout

Race the graph callback against the remaining absolute agent deadline through
the injected `Clock`. Timeout returns the authored Agent timeout error, cancels
callback work, reaps adapters exactly once, and skips artifact/verifier work.
Preparation consumes the same deadline; no fresh timeout begins. SimClock tests
prove timeout, cancellation, cleanup, and an on-time success path.

## E04: Stop tools on fail-fast abort

Pre-aborted tools return without dispatch. In-flight tools race dispatch against
abort and publish neither output nor successors after abort. Completed tools
are not retroactively undone; resilient policy is unchanged.

## E05: Settle closed-worker credits durably

Coordinator-local guaranteed storage shall retain synthetic returns that cannot
enter the bounded worker-return channel. Every enrollment UUID settles exactly
once; no unbounded worker-return channel is introduced. A forced-full-channel
test proves all returns, zero outstanding enrollment, and idle completion.

## E06: Make sidecar lifecycle cleanup exhaustive

Track attempted/started sidecars. After start failure, clean every relevant
sidecar once; after finish failure, still attempt every started sibling finish.
Return the primary error with cleanup context. Keep serial ordering/timestamps;
do not add concurrent lifecycle calls. Tests cover start failure, finish
failure, and normal one-start/one-finish behavior.

## E07: Preserve cancellation through global-push accounting

Carry a typed cancellation terminal outcome across the worker-credit seam and
map it to `ReplayTerminalStatus::Cancelled`. It settles accounting without
latching `OnFailure::Abort`; genuine sink errors remain Failed and do latch it.

## E08: Implement declared NativeGraph terminal outputs

Validated terminal-output declarations are allowed. Completed stage channels
freeze through an execution-owned content-addressed segment seam and become
resolvable handles in the completion supplement. Missing declared output is a
typed observation error. Supplements contain exactly declared outputs and never
embed raw terminal values or undeclared channels.

## E09: Enforce one dataset before adaptation

Protocol-v2 rejects zero or more than one datasets with the exactly-one
diagnostic before adaptation. The existing one-dataset projection and selected
factory are unchanged. Dataset composition is out of scope.

## E10: Unify Docker verifier transaction handling

Route ordinary single-step, NativeGraph, and multi-step verification through one
shared verifier transaction/helper. In shared mode cleanup of `/tests` and
`/logs/verifier` runs after success and verifier failure; cleanup failure joins
the primary error. Separate verification does not run that cleanup. Preserve
isolation, deadlines, staged artifacts, reward parsing, and container cleanup.

## E11: Resolve Dynosim input before process defaults

Resolve execute input exactly once into resolved-run bytes before selecting
Dynosim process defaults, then execute those same bytes. Authoring-envelope and
bare-resolved Dynosim input select equivalent defaults; HTTP does not; explicit
environment values win. Malformed input retains its typed protocol failure.

## Verification

Each task requires a demonstrated RED-to-GREEN cycle, focused tests using
`CARGO_TARGET_DIR` below `/mnt/4tb`, a Graham review with no open blocker, and a
commit.
