<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RNG derivation

## Purpose

`aiperf_runtime::rng` is the hash-derived randomness substrate. A component
anywhere in the tree gets its own reproducible, statistically independent stream
by naming itself; no seed is threaded through call sites. The deterministic vs
non-deterministic choice is made once, at the root.

## Built

### Derivation

`RngRoot(Option<u64>)` derives child streams by hashing `"{root}:{identifier}"`
(and indexed variants `"{root}:{identifier}:{index}"`) with BLAKE3, streaming the
parts directly without allocating a formatted namespace string. `RngRoot::derive`
returns a `RandomGenerator`. A child seed depends only on `(root, identifier)`, so
it is order-independent (adding, reordering, or parallelizing other streams leaves
every other stream unchanged), hierarchical for free (dotted identifiers are just
strings), and stable across process restarts and machines. `root = None` yields
non-deterministic streams from OS entropy; a set root makes the run reproducible.

### Generators and sampling

`HashIdRandomGenerator` re-seeds per datum for parallel trace synthesis, so
independent workers decoding the same `hash_id` produce identical tokens with zero
coordination. Internal reproducibility uses deterministic `Pcg64`/`NumpyPcg64` plus
`rand_distr` — not cross-language byte parity. The module provides generic sampler
seams, five sampling distributions, and sequence distributions, and canonical
namespace constants.

### Consumers

Dataset composition and samplers, ancillary timing, and graph
phase/arrival/node-cancellation/worker synthesis streams consume this substrate.

## Future requirements

- Broader non-graph request-scheduler integration of the derivation substrate.

## Source anchors

- `rust/runtime/src/rng/` (`derive.rs` `RngRoot`, `generator.rs`
  `RandomGenerator`, `hash_id.rs` `HashIdRandomGenerator`, `numpy_pcg64.rs`,
  `numpy_generator.rs`, `dist.rs`, `namespace.rs`).
