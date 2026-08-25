# Commit 009 — `844efe1b36` — full synthesis prefix blocks

## Upstream intent

`844efe1b36` prevents a synthetic session's L2 content from sharing the final
L1/L1.5 cache block. L1 and L1.5 each consume an independently rounded-up
number of blocks. The session boundary must therefore be
`(ceil(L1 / block_size) + ceil(L1.5 / block_size)) * block_size`, rather than
the unrounded sum of their configured token lengths. Otherwise a shared hash
ID can be the partial final block for one row and a complete block for another,
which makes trace reconstruction reject the dataset.

## Native comparison

**Status: applicable and ported.** Before this port,
`rust/cli/src/synthesize/prefix.rs` already allocated L1 and L1.5 as whole
blocks (`l1_blocks`, `l15_blocks`, and `prefix_blocks`), but
`rust/cli/src/synthesize/synth.rs` initialized `fixed_prefix` from
`layer1_tokens + layer1_5_tokens`. With a 64-token block, L1=1000 and L1.5=500
allocate 16 + 8 = 24 blocks (1536 tokens), while the sampler started L2 at
token 1500. A small L2 draw could consequently make the final shared ID (block
23) a row's partial final block.

The native correction is to expose the allocator's rounded prefix-token span
and use it when sampling the initial context. This keeps session content past
the shared prefix, preserves the existing seeded draw order, and requires no
wire-format change.

## Required verification

- A focused native synthesizer regression with non-multiple L1/L1.5 sizes and
  a one-token L2 draw must show the first turn begins at the rounded shared
  boundary plus one token, not at the raw token sum plus one.
- The synthesis CLI tests must pass after the correction.
- Refresh the native byte-parity fixtures from the upstream Python oracle:
  their default L1.5 span is not block aligned, so their old bytes encode the
  bug and must not be retained as a compatibility contract.
- Re-run the existing native Mooncake replay coverage to confirm generated
  `hash_ids` remain consumable by the Rust trace loader.

## Port closure

The actual upstream commit was incorporated by non-fast-forward merge
`cb1d016398`. Native implementation commits `ca2866de374` and `2d93736053f`
expose the allocator-owned rounded span, consume it in synthesis, refresh all
three Python-oracle parity fixtures, and prove the 1537-token / 25-ID boundary
including distinct shared and session-owned IDs.

Validation passed: focused native synthesis (1 test), native synth parity (3
tests), Mooncake replay (19 tests), and the upstream Python roundtrip/profile
coverage (10 tests, 3 intentionally deselected). Python checks explicitly used
`PYTHONPATH=/mnt/4tb/aiperf-origin-port-009/src`: the environment's editable
install otherwise imported the concurrent shared checkout, where this port was
not present, and produced a false 512-versus-96 shared-block failure. The
worktree-pinned focused regression also passed. Scoped `rustfmt --check` and
`git diff --check` passed; workspace-wide formatting remains blocked solely by
unrelated `runtime/src/engine/sidecar_input.rs` churn.

Late remediation: JSON-supplied numeric bounds are now checked before prefix
layout construction, and the synthesizer returns an error if its fixed shared
prefix cannot be extended by L2. Focused boundaries cover
`block_size=i64::MAX` with one L1 and one L1.5 token, plus the reachable valid
`i64::MAX` fixed-prefix/L2 addition; neither path panics or wraps.

## Native integration closure

`rust/e2e-tests/tests/test_synthesized_mooncake_trace.rs` now covers the
end-user command boundary omitted by the earlier unit and fixture checks. It
writes both a generic block-aligned synthesis configuration and the upstream
partial-prefix regression configuration (64-token blocks, L1=1000, L1.5=500,
small L2 tails, three groups, seed 42), invokes native
`aiperf synthesize agentic-code`, and replays the resulting Mooncake JSONL
through native `aiperf profile` against the in-process Rust mock server.
It asserts the request count, positive request and completion lengths, summary
and CSV artifacts, absence of an unprojectable `inputs.json`, and the complete
set of synthesized session identities.

The native request-budget scheduler can redistribute turn counts among the
selected sessions, so the Rust assertion intentionally compares the stable
session-id set rather than assigning a source turn count to one particular
session. Per-record sequence lengths are read from the native JSONL metrics
wrapper (`metrics.<name>.value`). Those are harness/artifact-shape adaptations,
not a product behavior gap.

The controlled RED run temporarily restored the former raw 1500-token prefix
calculation. Profile then refused the generated trace with `hash id 223 was
first materialized with 34 tokens but is now requested with 56`; restoring
`allocator.prefix_tokens()` made the full native E2E pass again.
