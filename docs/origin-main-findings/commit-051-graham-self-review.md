# Commit 051 Graham self-review

## Review scope

- Exact implementation range: `7520bbefccca56c75d660ac07e18bb765f7d18cb..0f3a078bc4`.
- Upstream semantic source: exact `324bb05773b3f99743c6516018f3c30cfe33de0b`.
- Review method: two complete passes over every changed production hunk and its
  nearest caller/consumer, followed by test-only and documentation passes.
- Standards: error handling, tracing and log levels, ownership/clones, locking,
  async and clock correctness, bounds, public wire compatibility, and minimal
  diff surface from the Graham review rubric.

## Semantic trace

The authored boolean crosses CLI/YAML, typed config, resolver, protocol-v2, and
prepared endpoint state without acquiring a second source of truth. Validation
fails closed unless chat, streaming, and server token counts are all effective.
The body formatter uses object `entry().or_insert()` semantics so authored
values remain authoritative.

The HTTP reducer retains the cumulative completion count on the first meaningful
reasoning/content frame only when the prepared endpoint opts in. The optional
value crosses the transport-neutral observer seam and compact record ingest
without synchronization. One shared checked helper computes the decode-token
divisor for both primary and adaptive metrics; missing or inconsistent endpoint
counts preserve the legacy divisor.

The mock server keeps `continuous_usage_stats` independent from `include_usage`,
uses the same renderer for fast and timed paths, and retains pre-error cumulative
usage without synthesizing a terminal frame. Request capture is disabled by
default, bounded by configured entry count and an 8 MiB body limit, and locks
only after body collection has completed. No lock is held across `.await`.

## Findings and resolutions

### Important — real HTTP route evidence was incomplete

The renderer-level tests covered chat, completions, timed, reasoning, absent,
false, terminal, and error semantics, but the real Axum routes did not directly
assert `/v1/completions` or raw mid-stream error SSE. This left handler routing
and byte framing below the intended evidence bar.

Resolved in `952170712b`: a real-server `/v1/completions` test proves cumulative
counts `3,4,5,6`, no terminal usage when `include_usage=false`, and `[DONE]`; a
real-server chat error test proves cumulative usage on the emitted content frame,
the raw `event: error` marker, no `[DONE]`, and no empty-choice terminal frame.

### Important — a typed CLI fixture did not initialize the new policy

The broad CLI build supplied a compile-failure RED after a pre-existing unit
fixture constructed `Endpoint` exhaustively without the new field. Resolved in
`0f3a078bc4` by explicitly initializing the fixture to the public default
`false`; its focused control-hook tests then passed 4/4.

### No unresolved Critical, Important, or Minor findings

The second pass found no production `unwrap`/`expect`, lock-across-await,
unbounded channel, redundant synchronization, non-Clock timing, hot-path log,
or unnecessary clone introduced by this range. Existing mock serialization
assumptions moved between helpers but were not newly introduced on the runtime
hot path. The mock and exact-oracle capture facilities are test-target-only and
opt-in.

## Verification observations

- `git diff --check 7520bbefcc..HEAD`: pass.
- Exact merge parents: first `5414b19168`, second exact upstream `324bb05773`;
  merge tree equals first-parent tree (`bee74411...`).
- Exact Python-oracle/native E2E: 14 passed in the focused run before this review.
- Direct real HTTP route controls: 2 passed.
- Full mock package excluding one proven inherited test: 175 unit, 7 accuracy,
  3 balancer, 4 gRPC, 52 HTTP integration, 3 TLS, and 15 WebSocket passed.
- Full unfiltered mock run's sole failure was
  `empty_prompt_yields_zero_completion`; exact detached pre-port merge
  `7520bbefcc` reproduced the same `1 != 0` result.
- Default runtime full run: 1810 passed, 7 ignored, 2 unrelated failures. The
  cellular publication deadline passed immediately on focused rerun. The report
  golden expects package version `0.0.0` while the unchanged current package
  emits `0.12.0`; neither failing source file differs in this review range.
- Focused changed runtime surfaces: metrics 2, endpoint 3, config 3, reduction
  2, and adaptive 1 all passed.
- Engine-feature runtime: 2369 passed and 7 ignored; six unrelated failures are
  the known deadline/report pair, two absent recorded-agent fixtures, and two
  existing registry/bootstrap expectations in untouched files.
- CLI: the focused repaired fixture passed 4 and dedicated per-chunk authoring
  passed 2. The broad run built and passed all 261 library tests before an
  unrelated graph-help golden omitted the existing `tracelab` input.
- Fresh pinned-binary exact-upstream/native E2E: 14 passed; no oracle worktree
  remained afterward.
- Clippy over every target in all four changed crates completed successfully;
  its warnings are inherited and outside the port's changed hunks.
- `cargo fmt --all -- --check` and the final `git diff --check` passed.

## Disposition

Self-review disposition: approve. Independent Graham review subsequently
approved `7520bbefcc..8fa60b9800` after two passes with no blocking, Important,
or style findings.
