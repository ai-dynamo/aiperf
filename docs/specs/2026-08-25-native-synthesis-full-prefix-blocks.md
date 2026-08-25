# Native synthesis full-prefix block boundary

## Goal

Make native `aiperf synthesize agentic-code` reserve the complete L1 and L1.5
shared cache prefix before it places session-specific L2 content.

## Problem

The prefix allocator independently rounds L1 and L1.5 to whole cache blocks,
because both layers produce reusable hash IDs. The session synthesizer samples
its initial context from the raw configured token sum instead. When either
layer is not an exact multiple of `block_size`, this lets L2 begin in the final
shared block. That block's ID can then represent a short terminal block in one
serialized row and a full block in another, violating the trace loader's
one-hash-ID/one-block-size invariant.

## Design

`PrefixAllocator` owns the layout arithmetic, so it will provide a
`prefix_tokens()` accessor equal to `prefix_blocks * block_size`. The
`SessionSynthesizer` will derive `fixed_prefix` from that accessor rather than
duplicating the raw token arithmetic. It will continue to sample the same L2
distribution and preserve all existing random-draw order; only the deterministic
base added to that draw changes when a configured layer has a partial block.

No dataset schema, CLI flag, hash-ID allocation range, or runtime loader API
changes. The emitted first turn will always extend past the entire rounded
shared prefix before assigning session-owned content.

The native `synthesize_parity` fixtures encode the old pre-fix behavior. The
default L1.5 span is not block-aligned, so the corrected native output is
intentionally different. Refresh those fixtures from the Python implementation
after merging the upstream commit, then verify native output is byte-identical
to that oracle for every existing seed/session-count case.

## Acceptance criteria

1. For block size 64, L1=1000, L1.5=500, and a deterministic one-token L2
   draw, first-turn `input_length` is 1537, not 1501.
2. The first 24 generated hash IDs remain the shared L1/L1.5 prefix, and the
   session's first token belongs after that prefix.
3. Each refreshed native parity fixture is byte-identical to the upstream
   Python synthesizer for its existing seed/session-count case.
4. The native synthesis and Mooncake trace replay test coverage passes.

## Non-goals

- Changing hash-ID formats or allocation ranges.
- Altering L2 sampling distributions or random-draw order.
- Changing Python synthesis behavior beyond the merged upstream commit.
