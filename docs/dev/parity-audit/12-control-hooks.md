<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Control hooks parity audit

Python reference side is `origin/main` @ `bc359bf8fd`
("fix(control-hooks): stop reset_kv_cache/server_profiler timeouts inheriting 6h
endpoint.timeout (#1332)"). All three control-hooks Python files
(`src/aiperf/config/control_hooks.py`, `src/aiperf/common/control_hooks.py`,
`src/aiperf/common/control_plane_http.py`) are byte-identical between the branch
working tree and that revision, so the working tree was read directly and no
separate baseline checkout was used.

## Summary

The hand-match holds on every numeric and string value that #1332 introduced.
Most importantly, item 1 — the regression the upstream fix targeted — is
correct in Rust: an unset `timeout_seconds` resolves to a hard-coded 30 s and
the endpoint's 6-hour inference timeout cannot reach a control POST, because the
control-plane layer enforces `min(absolute_deadline, cancellation_deadline)` with
its own `Clock` sleep rather than deferring to the HTTP client's
`total_timeout_ns`. The 60 s retry budget, the exact `{409, 423, 429, 503}`
retryable set, total-budget-vs-per-attempt retry semantics, the 1 s → 8 s
doubling backoff, the three default paths, and the abort-vs-warn failure policy
all match precisely.

The divergences are not in the ported constants but in the surrounding request
and multi-origin plumbing, which #1332 did not touch and which the porting agent
appears not to have carried over. The one that applied to every single-origin run
— control-plane POSTs carrying no `Authorization` header and no
`endpoint.headers` — is fixed; finding 1 records the fix. The remaining three are
gated on a configuration with more than one endpoint URL: no origin
de-duplication, no reverse-order rollback after a partial profiler start, and a
profiler stop that aborts at the first failing origin instead of attempting all
of them. Three open P1 findings and one P2; no P0.

## Value-by-value verification table

| # | Item | Python value + `path:line` | Rust value + `path:line` | Match? |
|---|---|---|---|---|
| 1 | Default control-hook timeout | `30.0` — `src/aiperf/config/control_hooks.py:18`; applied `src/aiperf/common/control_hooks.py:94-103` | `30_000_000_000` ns — `rust/runtime/src/engine/control_hooks.rs:30`; applied `:473-483` | Yes |
| 1b | Independent of `endpoint.timeout` (6 h) | Fallback is the constant, never `endpoint.timeout` — `src/aiperf/common/control_hooks.py:97,102`; asserted `tests/unit/common/test_control_hooks.py:44-64` | `resolve_timeout_ns(None)` → constant, never `client.total_timeout_ns` — `rust/runtime/src/engine/control_hooks.rs:473-477`; enforced by `min()` + Clock sleep at `rust/runtime/src/engine/control_plane_http.rs:895-909`; asserted `rust/runtime/src/engine/control_hooks.rs:997-1014` | Yes |
| 2 | Default reset retry budget | `60.0` — `src/aiperf/config/control_hooks.py:25`; applied `src/aiperf/common/control_hooks.py:104-109` | `60_000_000_000` ns — `rust/runtime/src/engine/control_hooks.rs:31`; applied `:537-542` | Yes |
| 3 | Retryable status set | `frozenset({409, 423, 429, 503})` — `src/aiperf/config/control_hooks.py:37`; used `src/aiperf/common/control_hooks.py:154-157` | `[409, 423, 429, 503]` — `rust/runtime/src/engine/control_hooks.rs:41`; used `:628,651-655` | Yes, exactly |
| 3b | All other non-2xx never retried | `src/aiperf/common/control_hooks.py:154-159`; asserted `tests/unit/common/test_control_hooks.py:306-326` | `:628` (`contains` is false) → `:668` returns `Err`; asserted `rust/runtime/src/engine/control_hooks.rs:1016-1046` (`400` → 1 attempt) | Yes |
| 4 | Retry budget kind | Total wall-clock: `deadline = time.monotonic() + max_retry_seconds` set once per URL — `src/aiperf/common/control_hooks.py:147` | Total wall-clock: `retry_deadline_ns = clock.now_ns() + max_retry_ns` set once per handle — `rust/runtime/src/engine/control_hooks.rs:659` | Yes |
| 4b | Give-up predicate | `time.monotonic() + backoff >= deadline` — `src/aiperf/common/control_hooks.py:158` | `clock.now_ns() + backoff_ns >= retry_deadline_ns` — `:669` | Yes |
| 4c | Backoff schedule | `1.0` initial, `×2.0`, cap `8.0` — `src/aiperf/config/control_hooks.py:31,33,35`; `src/aiperf/common/control_hooks.py:148,160-164` | `1e9` initial, `×2`, cap `8e9` — `rust/runtime/src/engine/control_hooks.rs:32,33,34`; `:660,680-683` | Yes |
| 4d | Per-POST timeout across retries | Fresh `timeout_s` passed to each `control_plane_post` — `src/aiperf/common/control_hooks.py:151` | Fresh `absolute_deadline_ns = now + timeout_ns` per attempt — `rust/runtime/src/engine/control_hooks.rs:601` | Yes (retry resets, does not consume) |
| 4e | Profiler hooks get no retry | Direct `control_plane_post`, no retry wrapper — `src/aiperf/common/control_hooks.py:193,204,238` | `max_retry_ns = 0` → empty retryable set + immediate deadline — `rust/runtime/src/engine/control_hooks.rs:343,366,651-655` | Yes |
| 5 | `timeout_seconds` bound | `default=None, gt=0`, finite — `src/aiperf/config/control_hooks.py:51-59, 86-94` | Finite and `> 0` — `rust/runtime/src/engine/control_hooks.rs:498-509` | Value bound yes; rejection *stage* differs (see finding 5) |
| 5b | `max_retry_seconds` bound | `default=None, ge=0`, finite — `src/aiperf/config/control_hooks.py:60-72` | Finite and `>= 0` — `rust/runtime/src/engine/control_hooks.rs:511-522` | Value bound yes; rejection *stage* differs (see finding 5) |
| 6 | `extra="forbid"` | `src/aiperf/config/control_hooks.py:49, 84` | `#[serde(deny_unknown_fields)]` on `ResetKvCacheYaml` / `ServerProfilerYaml` — `rust/cli/src/yaml.rs:1063, 1074` | Yes |
| 6b | `false \| true \| object` shorthand | `parse_enabled_or_config` — `src/aiperf/config/control_hooks.py:128-144`; wired `src/aiperf/config/endpoint.py:458-471` | `EnabledOrConfig<…>` — `rust/cli/src/yaml.rs:1140-1143`, mapped `:1870-1881`; tested `:3847` | Yes |
| 7 | Default reset path | `/reset_prefix_cache` — `src/aiperf/config/control_hooks.py:15` | `/reset_prefix_cache` — `rust/runtime/src/engine/control_hooks.rs:42` | Yes |
| 7b | Default profiler start path | `/start_profile` — `src/aiperf/config/control_hooks.py:16` | `/start_profile` — `rust/runtime/src/engine/control_hooks.rs:43` | Yes |
| 7c | Default profiler stop path | `/stop_profile` — `src/aiperf/config/control_hooks.py:17` | `/stop_profile` — `rust/runtime/src/engine/control_hooks.rs:44` | Yes |
| 7d | Path × base-URL join | Origin (`scheme://netloc`, path/query stripped) + leading-slash path — `src/aiperf/common/control_hooks.py:32-40` | `set_path("/")`, query and fragment cleared, then `url.set_path(path)` — `rust/runtime/src/engine/control_hooks.rs:392-403`; `rust/runtime/src/engine/control_plane_http.rs:1087-1092` | Yes for one URL; see finding 4 for multi-URL |
| 8 | reset_kv_cache failure → abort | Raises out of `maybe_reset_kv_cache_before_run`, aborting before services start — `src/aiperf/common/control_hooks.py:176-182`; `src/aiperf/cli_runner/_single_run.py:42-50` | `?` out of `run_reset_kv_cache_before_run`, aborting before child launch — `rust/cli/src/control_hooks.rs:73-82` | Yes |
| 8b | profiler start failure → abort | Raises → `_execute_phases` cancels and re-raises — `src/aiperf/timing/phase_orchestrator.py:88, 458-461` | Sidecar `start` error → `PhaseExecutionError` from phase setup — `rust/runtime/src/engine/server_profiler.rs:66-70`; `rust/runtime/src/phase_runtime.rs:217-238` | Yes |
| 8c | profiler stop failure → warn, run still succeeds | `_stop_server_profiler_warn_only` — `src/aiperf/timing/phase_orchestrator.py:49-59, 103` | `tracing::warn!` then `Ok(())` — `rust/runtime/src/engine/server_profiler.rs:96-102`; run-shutdown `force_stop` warn — `rust/runtime/src/engine/execute/sidecars.rs:309-316` | Yes |
| 9 | Profiler fires on profiling phases only | `phase == CreditPhase.PROFILING` gate — `src/aiperf/timing/phase_orchestrator.py:79-83` | `!phase.common().exclude_from_results` gate (warmup sets it `true`) — `rust/runtime/src/engine/execute/compose_sidecars.rs:22, 31-36`; `rust/runtime/src/engine/online_execution.rs:1696`; sharded arm hard-codes `"profiling"` — `compose_sidecars.rs:732-737` | Yes |
| 9b | Start once / stop after last owner | Owner set, start when empty, stop when drained — `src/aiperf/timing/phase_orchestrator.py:521-544` | `Cell<usize>` owner count with same predicate — `rust/runtime/src/engine/control_hooks.rs:157-208`; tested `:920-961` | Yes |

## Findings

### 1. Control-plane POSTs carry the endpoint's auth headers and `endpoint.headers` (fixed)

**Severity: P1 — fixed**

**Python evidence.** Every control-plane POST is given the endpoint's auth
headers. `src/aiperf/cli_runner/_single_run.py:49` calls
`auth_headers_for_endpoint(endpoint)` and passes the result into
`run_reset_kv_cache`; the phase orchestrator threads the same dict through as
`control_headers` (`src/aiperf/timing/phase_orchestrator.py:184`) into
`start_server_profiler` / `stop_server_profiler`. That helper copies
`endpoint.headers` verbatim and adds scheme-appropriate credentials —
`Authorization: Bearer <api_key>`, or `x-api-key` plus `anthropic-version` for
the Messages endpoint (`src/aiperf/common/endpoint_auth.py:22-38`, whose comment
explicitly documents that custom headers pass through because "control-plane and
readiness endpoints need [them] just as much as inference does"). The headers
reach the wire at `src/aiperf/common/control_plane_http.py:65`.

**Original Rust divergence.** `control_hooks.rs` built every control-plane
profile with `ControlPlaneCredentialReference::None`, and the profile-value DTO
`ControlHookProfileValue` deserialized neither `headers` nor `api_key`, so those
fields were unavailable to the hook path. `ClientConfig`
(`rust/runtime/src/transport/http/config/defaults.rs:160-220`) has no header
field, so cloning the endpoint client carried nothing either. The transport sent
only `Accept` and `Accept-Encoding`; the sole credential path was a
secrets-resolver-backed `BearerProvider`, which control hooks never selected —
and which the https-only credential guard would in any case have refused on an
`http://` control URL. With `--api-key` set, or with a gateway requiring a custom
`endpoint.headers` entry, `reset_kv_cache` received 401/403 — outside the
retryable set — so the run aborted before warmup, and warn-only `stop_profile`
failures left the server profiling after a run that still reported success.

**Rust resolution.** The dialect auth rules are now single-sourced and reused by
the hook path rather than restated:
`apply_bearer_auth_header` (`rust/runtime/src/endpoints/implementation.rs:2096`)
carries the `Authorization: Bearer` rule for the default dialect, and
`apply_messages_auth_headers` (`rust/runtime/src/endpoints/anthropic.rs:56`)
carries the Messages `x-api-key` hard-assign plus the `anthropic-version`
`setdefault`. Both are called from the dialects' own `format_headers` and from
`auth_headers_for_endpoint` (`implementation.rs:2119`), the out-of-band header
builder that mirrors `endpoint_auth.py`: authored `endpoint.headers` first, then
the endpoint-type-selected credential rule. It deliberately omits `Content-Type`,
matching the Python helper; the control-plane transport already defaults it.

`control_hook_request_headers` (`rust/runtime/src/engine/control_hooks.rs:367`)
resolves that set once per prepare and installs it on the validated profile
through `ValidatedControlPlaneProfile::with_request_headers`
(`rust/runtime/src/engine/control_plane_http.rs:610`), a validating builder that
keeps the seven-parameter constructor untouched. The carrier is
`ResolvedRequestHeaders` (`control_plane_http.rs:173`), which validates header
names as RFC 7230 tokens and rejects control characters in values, and whose
`Debug` prints names only — values never appear in `Debug`, tracing, or error
text, matching `ResolvedSecret`'s discipline. `NativeControlPlaneHttp::execute`
applies them after `Accept`/`Accept-Encoding` and before the provider-resolved
`Authorization`, so an explicitly configured deployment credential still wins
(`control_plane_http.rs:962`). Headers live on the profile rather than on
`ControlPlaneRequest` because they are constant for an origin, not per-request.

The https-only credential guard was deliberately **not** extended to these
headers: they are the same authored credentials the data plane already sends to
this origin, and refusing them on a plaintext endpoint would disable control
hooks against a local token-protected server where Python succeeds.

Both entry points are wired: the typed path reads `profile.endpoint_id` and the
already-decoded `RawEndpointConfig`, and the JSON path widens
`ControlHookProfileValue` (`control_hooks.rs:237`) with `api_key` and `headers` —
fields the endpoint-profile payload it re-decodes already carried for the
inference transport, so nothing new is exposed.

**Coverage.** Unit tests in `control_hooks.rs` assert the bearer set, the
Messages `x-api-key` + `anthropic-version` set, authored-header passthrough with
`api_key` overriding a preconfigured `x-api-key`, the no-credential case sending
no `Authorization`, and the JSON path carrying both fields.
`rust/e2e-tests/tests/test_control_hook_auth.rs` reads mock-server request
captures for `/reset_prefix_cache`, `/start_profile`, and `/stop_profile` to
prove the bearer and authored headers reach the wire, and that a credential-free
run sends no `Authorization`.

### 2. Multi-origin profiler stop aborts at the first failing origin instead of attempting all and aggregating

**Severity: P1**

**Python evidence.** `stop_server_profiler`
(`src/aiperf/common/control_hooks.py:224-256`) catches
`ControlPlaneHttpError` per URL into a `failures` list, continues the loop over
every remaining origin, and only then raises one aggregated error naming the
count. Its docstring and
`tests/unit/common/test_control_hooks.py:476-512` pin this: with origins A, B, C
where A and C fail, all three are still attempted and the message names both
failures.

**Rust evidence.** `execute_control_hook`
(`rust/runtime/src/engine/control_hooks.rs:658-685`) iterates the handles and, on
a non-retryable failure, executes `return Err(failure.error)` at `:671` from
inside the per-handle loop, so handles after the failing one are never invoked.
`stop_server_profiler` (`:350-370`) routes through this same function with
`max_retry_ns = 0`.

**Observable user impact.** With more than one endpoint origin, a single failing
`stop_profile` (a restarting replica, a transient 500) leaves the remaining
origins profiling indefinitely after the benchmark ends — continuing to burn
CPU and write profiler traces on those servers. The run still reports success
because the stop path is warn-only, so nothing tells the user which origins were
left running.

**Confidence: high** for the control flow. Requires ≥2 distinct endpoint origins.

### 3. Partial profiler start is not rolled back on already-started origins

**Severity: P1**

**Python evidence.** `start_server_profiler`
(`src/aiperf/common/control_hooks.py:185-221`) tracks each successfully started
URL and, in a `except BaseException` handler, POSTs `stop_path` in reverse order
for exactly the origins that started, logs any cleanup failure, then re-raises
the original error unchanged. Pinned by
`tests/unit/common/test_control_hooks.py:423-472` (start fails on B → A is
stopped) and `:556-590` (the same holds under `CancelledError`).

**Rust evidence.** `start_server_profiler`
(`rust/runtime/src/engine/control_hooks.rs:327-347`) delegates to
`execute_control_hook`, which has no rollback path — it returns `Err` at `:671`
with no stop POSTs issued. The coordinator cannot compensate either:
`ServerProfilerCoordinator::acquire` (`:167-180`) propagates the start error with
`?` at `:172` *before* incrementing `owners` at `:180`, so `owners` stays `0` and
both `release` (`:183-197`) and the run-shutdown `force_stop` (`:200-208`, called
from `rust/runtime/src/engine/execute/sidecars.rs:309-310`) short-circuit without
stopping anything.

**Observable user impact.** With more than one endpoint origin, a profiler start
that succeeds on origin A and fails on origin B aborts the run (correctly, and
matching Python) but leaves A profiling with no stop ever issued — the process
exits, so nothing will ever stop it. Python leaves the servers clean.

**Confidence: high.** Requires ≥2 distinct endpoint origins.

### 4. Endpoint origins are not de-duplicated, so a shared origin receives duplicate control POSTs

**Severity: P1**

**Python evidence.** `unique_endpoint_origins`
(`src/aiperf/common/control_hooks.py:61-71`) reduces the URL list to
order-preserving unique `scheme://netloc` origins, and
`prepare_endpoint_control_hooks` builds hook URLs from that reduced list
(`:78, 112-126`). `tests/unit/common/test_control_hooks.py:89-113` asserts
exactly this: three URLs across two hosts collapse to two reset URLs, two start
URLs, and two stop URLs.

**Rust evidence.** `prepare_handles`
(`rust/runtime/src/engine/control_hooks.rs:372-390`) loops over
`endpoint_urls` and unconditionally pushes one handle and one target URL per
input URL, with no `seen` set and no comparison of the derived base URLs. Every
downstream execution iterates that full handle list
(`:657-658`), so the request count equals the URL count.

**Observable user impact.** A common multi-path configuration such as
`urls: ["http://host:8000/v1/chat/completions", "http://host:8000/v1/completions"]`
sends two `POST /reset_prefix_cache`, two `POST /start_profile`, and two
`POST /stop_profile` to the same server where Python sends one of each. A second
`start_profile` on a server that rejects an already-running profiler returns a
non-2xx that is outside the retryable set, aborting the run; on a server that
accepts it, the two stops may not balance the two starts. The duplicate reset is
merely wasteful. `ControlHookOutcome.request_count` is inflated correspondingly.

**Confidence: high** for the missing de-duplication; **medium** for the specific
server reaction to a repeated `start_profile`, which is server-dependent.

### 5. Invalid `timeout_seconds: 0` and out-of-range `max_retry_seconds` are rejected at run time rather than at config-validation time

**Severity: P2**

**Python evidence.** The bounds live on the Pydantic fields —
`timeout_seconds` is `gt=0` (`src/aiperf/config/control_hooks.py:53-58` and
`:88-93`) and `max_retry_seconds` is `ge=0` (`:62-71`) — so `timeout_seconds: 0`
and `max_retry_seconds: -1` are rejected during model validation, which means
`aiperf config validate` fails on them.

**Rust evidence.** The config-layer check
`validate_control_hook_timeout` (`rust/runtime/src/endpoints/config.rs:596-605`)
only rejects non-finite or `value < 0.0`, so it **accepts** `timeout_seconds: 0`;
and `validate_against` (`:429-449`) never inspects `max_retry_seconds` at all.
The real bounds are enforced later, at hook-preparation time, by `seconds_to_ns`
(`rust/runtime/src/engine/control_hooks.rs:498-509`, finite and `> 0`) and
`nonnegative_seconds_to_ns` (`:511-522`, finite and `>= 0`).

**Observable user impact.** The accepted-value sets are identical, and both sides
refuse loudly — the difference is only *when*. `aiperf config validate` can pass a
config that a later `aiperf profile` rejects, and the diagnostic arrives from the
hook-preparation layer ("endpoint-local control hook timeout_seconds must be
finite and positive") rather than as a config-field error naming the path. Low
severity precisely because Rust refuses rather than silently substituting a
default.

**Confidence: high.**

## Checked and consistent

- **The 30 s vs 6 h timeout (item 1) is correct in Rust.** Beyond the constant
  matching, the enforcement path was traced: `attempt_control_request` sets
  `absolute_deadline_ns = clock.now_ns() + timeout_ns`
  (`rust/runtime/src/engine/control_hooks.rs:601`) and
  `NativeControlPlaneHttp::execute` takes
  `absolute_deadline_ns.min(cancellation.deadline_ns())` and races the request
  against its own `Clock` sleep (`control_plane_http.rs:895-909`). The 6-hour
  `total_timeout_ns` cloned into the control-plane `ClientConfig`
  (`control_hooks.rs:409-410, 448`) can only ever be the looser of the two, so it
  cannot extend a control POST past 30 s. Rust's own regression test
  (`control_hooks.rs:997-1014`) pins a 21 600 s client timeout against the 30 s
  constant, mirroring `tests/unit/common/test_control_hooks.py:44-64`.
- **Retry counting.** Both sides charge the retry budget for time spent inside
  failed attempts (the deadline is absolute, and both `time.monotonic()` and
  `clock.now_ns()` advance during the POST), and both reset the per-attempt
  timeout on each retry.
- **`max_retry_seconds: 0`.** Python treats a 503 as retryable but immediately
  fails the `monotonic() + backoff >= deadline` test; Rust disables the retryable
  status set entirely when `max_retry_ns == 0`
  (`rust/runtime/src/engine/control_hooks.rs:651-655`). Different mechanism,
  identical observable outcome: exactly one attempt, then failure.
- **reset_kv_cache stops at the first failing origin** on both sides
  (`src/aiperf/common/control_hooks.py:176-182` vs
  `rust/runtime/src/engine/control_hooks.rs:671`), matching
  `tests/unit/common/test_control_hooks.py:162-186`.
- **Overlapping/seamless profiler ownership** is reference-counted identically;
  Rust's `Cell<usize>` (`control_hooks.rs:157-208`) reproduces Python's owner-set
  semantics (`phase_orchestrator.py:521-544`), including start-once and
  stop-after-last-drain, and Rust's terminal `force_stop`
  (`sidecars.rs:309-316`) covers Python's `cancel()`/`on_stop` warn-only stop
  (`phase_orchestrator.py:589-601, 616, 639`).
- **Relative-path validation.** Python's `require_relative_path`
  (`src/aiperf/config/control_hooks.py:113-125`) rejects `://` and any path not
  starting with `/`. Rust's `validate_origin_relative_path`
  (`rust/runtime/src/endpoints/config.rs:607-624`) rejects the same, plus
  protocol-relative `//` prefixes and whitespace, and
  `normalize_control_path` (`control_hooks.rs:485-496`) additionally rejects `?`
  and `#`. Strictly narrower, and it refuses loudly — noted, not a finding.
- **Transport gating.** Rust refuses control hooks outright for `websocket`,
  `dry_run`, and both dynosim transports
  (`rust/cli/src/control_hooks.rs:27-46`). This is a loud refusal with no Python
  counterpart; noted, not a finding.
- **Extra CLI surface.** Rust adds `--reset-kv-cache-max-retry-seconds`
  (`rust/cli/src/flags.rs:81-84`), which Python exposes only through YAML
  (`src/aiperf/config/flags/cli_config.py:273-317` has no such flag). A superset
  of authorable surface, not a behavior divergence.
- **Retryable-error classification.** Rust treats `Transport` and `Timeout`
  error kinds as retryable (`control_hooks.rs:616-619`, with `Connect`,
  `Protocol`, `Sse`, `Other` all mapping to `Transport` at
  `control_plane_http.rs:923-930`); Python marks `TimeoutError` and
  `aiohttp.ClientError` retryable (`control_plane_http.py:78-82`). Equivalent
  coverage of connect failures and timeouts.
- **Response bodies are never included in errors** on either side
  (`control_plane_http.py:70-75` drains without quoting;
  `control_hooks.rs:629-632` formats only the status).

## Unverified / needs runtime check

- **Whether a real server rejects a duplicate `POST /start_profile`** (finding 4).
  The doubled request count is certain from the code; the server-side consequence
  is not. Needs an end-to-end run with
  `urls: [http://host:8000/v1/chat/completions, http://host:8000/v1/completions]`
  and `server_profiler: true` against `aiperf-mock-server`, asserting the
  recorded request count per control path is 1.
- **The 30 s ceiling under a genuinely stalled control route.** Verified by code
  above, but no test on either side holds a control POST open past the deadline.
  Needs a mock-server route that sleeps ≥35 s on `/reset_prefix_cache` and an
  assertion that the run fails at roughly 30 s, not at `endpoint.timeout`.
- **Existing Rust e2e coverage does not reach the three open findings.**
  `rust/e2e-tests/tests/test_reset_kv_cache.rs` (three tests: once-per-sweep-cell,
  failure-aborts-before-warmup, cellular-once-on-controller) and
  `rust/e2e-tests/tests/test_server_profiler.rs` (three tests:
  starts-and-stops-around-profiling, stop-failure-preserves-successful-run,
  cellular-controller-owned) are all single-origin. Auth headers are now covered
  by `rust/e2e-tests/tests/test_control_hook_auth.rs`; multiple origins and
  timeout enforcement still have no Rust test.
- **Whether any deployed configuration relies on `endpoint.headers` reaching the
  control plane** (finding 1). The Python behavior is deliberate and documented in
  `docs/tutorials/benchmark-control-hooks.md`, but the breadth of real-world
  reliance is a product question, not a static one.
