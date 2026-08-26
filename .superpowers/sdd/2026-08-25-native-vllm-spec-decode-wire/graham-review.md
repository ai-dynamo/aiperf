# Graham self-review — native vLLM spec-decode wire

## Review boundary

- Base: `13fdae44306aa1c04d8ce7a9c71c5c92f53797fe`
- Reviewed implementation tip before this receipt: `08c43bef3c`
- Exact origin target: `810fd8bdd40a1c35b64d487b3b8487f0a71a0f6b`
- Provenance merge: `9fcd17c62e989fc35dd4358418350d458c52dbc5`

The provenance merge has first parent `c3c1f06f73ef18a1f8c423cedd49c7f2f9f8b542`,
second parent equal to the exact target above, and tree
`63a628a25b68733a84d58ff9399cfe68c11b635c`, equal to its first-parent tree.
No upstream Python tree or cherry-pick entered the native branch.

## Source and semantic audit

I reread the exact upstream endpoint extractor, chat and completions request
formatters, vLLM adapter, models, and translated upstream tests after the Rust
implementation was complete. The native path now has the same load-bearing
contract:

- only object-valued response-root `metrics.speculative_decoding` is captured;
- capture is independent of `choices`, including an empty-choice trailing usage
  frame, and the obsolete per-choice field is refused;
- the dense integer histogram is validated in full, width-checked when
  `num_spec_tokens` exists, zero-pruned into the sparse neutral map, and checked
  against aggregate and detailed counts;
- effective post-merge streaming chat and completions negotiate
  `stream_options.include_usage` independently of server token counting while
  preserving explicit author values;
- the mock emits the reviewed root wire in both endpoint families and both
  transport modes, with streaming metrics only on the requested usage frame;
  and
- the native process E2E omits both the manual stream-options escape hatch and
  server-token-count selection, so it exercises the production negotiation.

The final source audit found stale public reference/tutorial text and the old
#13 design's unmarked wire section. Commit `08c43bef3c` corrects those and its
documentation guard, license, spelling, and other commit hooks all passed.

## Graham passes

### Correctness and failure behavior

The parser uses checked length/index/sum/product arithmetic and keeps malformed
telemetry non-fatal through the existing structured-warning degradation path.
The empty-choice usage frame takes the generic decoder, so usage and metrics are
captured together without inventing a token/content event. Finish-reason and
tool-call semantics no longer depend on telemetry presence. No finding remains.

### Tracing and diagnostics

No log statement, log level, or hot-path tracing was added. Existing malformed
payload diagnostics remain one structured `warn!` with endpoint/request fields;
restricted-data behavior is unchanged. No finding remains.

### Ownership, allocation, and clones

No `Arc`, `Mutex`, channel, task, or clock handle was added. Ordinary chat token
chunks no longer deserialize an optional arbitrary `Value`; only the single
usage chunk enters generic JSON decoding. The dense-to-sparse conversion moves
the vector and allocates only the canonical `BTreeMap`. No new production clone
appears in the reviewed runtime diff. No finding remains.

### Async, concurrency, and timing

The change does not spawn work, block, hold a lock, or read wall/monotonic time.
It stays within the existing worker-local response-reduction boundary. No
finding remains.

### Diff and tests

The implementation changes only the vLLM wire adapter, OpenAI body negotiation,
typed fast-path cleanup, deterministic mock wire, and applicable unit,
transport, serializer, and native E2E coverage. A first review noted repetitive
chat/completions formatter setup; `6bc724e921` consolidated it into one helper.
Public docs were then corrected as described above. No finding remains.

## Verification receipts

All commands used `RUSTC_WRAPPER=sccache`,
`SCCACHE_DIR=/mnt/4tb/sccache-port052`, and
`CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port052` where Cargo was involved.

- parser RED: 5 expected failures of 8; GREEN: 8/8;
- request formatter RED: 3 expected failures of 20; GREEN: 20/20;
- typed chat codec: 6/6;
- real Hyper/SSE trailing-usage transport integration: 1/1;
- mock wire RED: 4 expected failures of 5; GREEN: 5/5;
- hand serializer versus serde byte parity: 1/1, including root metrics;
- endpoint/runtime focused slice: 54/54;
- full mock-server library: 166/166;
- native CLI build: passed;
- native real-process E2E: 15/15 across chat/completions and
  streaming/non-streaming, including detailed artifacts and no-stats omission;
- `cargo fmt --manifest-path rust/Cargo.toml --all -- --check`: passed;
- runtime+mock changed-scope Clippy with `--no-deps`: exit 0 (unrelated baseline
  warnings elsewhere in the repository remained);
- `git diff --check`: passed; and
- full `aiperf-runtime --features engine --lib`: 2,370 passed, 6 failed, 7
  ignored. The six failures are outside this port's changed files: two refer to
  missing recorded-agent fixtures, two expose unrelated registry/global-state
  assumptions, one is a global-push characterization count mismatch, and one
  expects version `0.0.0` while the worktree reports `0.12.0`. Every changed
  endpoint and transport test passed within this broad run. Per campaign
  direction, these concurrent/unrelated failures are recorded, not modified.

An earlier broad link attempt also encountered a concurrent `rust-lld` bus
error and produced `rust/core.<pid>` files. Only those exact generated files
were deleted at root's request. Subsequent CLI, focused, mock, and E2E builds
completed successfully.

## Verdict

**APPROVE for independent Graham review.** I found no unresolved correctness,
systems, performance, observability, or scope issue in the immutable
base-to-tip implementation. Campaign tracker closure remains intentionally
pending until root gives explicit independent approval.
