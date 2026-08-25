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

**Status: applicable.** `rust/cli/src/synthesize/prefix.rs` already allocates
L1 and L1.5 as whole blocks (`l1_blocks`, `l15_blocks`, and
`prefix_blocks`). But `rust/cli/src/synthesize/synth.rs` currently initializes
`fixed_prefix` from `layer1_tokens + layer1_5_tokens`. With a 64-token block,
L1=1000 and L1.5=500 allocate 16 + 8 = 24 blocks (1536 tokens), while the
current sampler starts L2 at token 1500. A small L2 draw can consequently make
the final shared ID (block 23) a row's partial final block.

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
