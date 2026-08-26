# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Origin #59 mmap conversation cache boundary

## Purpose

Record why origin/main `c9288da6c1` has no native implementation target while
preserving the rule that an upstream performance or concurrency fix is ported
whenever its affected mechanism exists in Rust.

## Built

The native dataset is an immutable in-memory `Dataset`: conversation storage
is `Arc<[Conversation]>`, lookup is an immutable `HashMap<SessionId, usize>`
followed by indexed borrowing, and worker materialization shares the lowered
`Arc<NativeDataset>`.  Stateful sampling and live multi-turn reply history are
kept worker-local; they are not dataset-file state.

Python's upstream change applies only to its `MemoryMapDatasetClient` data-file
cursor and its async wrapper.  Native has no data/index mmap files, no mmap
client store, no page-prefault control, and no executor dispatch surrounding a
conversation lookup.  The exact upstream object entered history as ancestry of
the #60 actual merge `f1d39ad583`.

## Requirements

1. Do not introduce an mmap cache, page-prefault environment variable, or
   executor hop merely to mimic an absent Python implementation.
2. Keep native conversations immutable after dataset construction and share
   them through existing `Arc` ownership; per-session mutable state remains
   worker-local.
3. If a native mmap-backed conversation source is introduced later, it must
   use position-free reads, define its page-population policy before timed
   traffic, and include a deterministic concurrent-reader regression test.

## Source anchors

- `runtime/src/dataset/runtime_dataset.rs`: frozen dataset construction and
  position lookup.
- `runtime/src/multiturn.rs`: source construction and worker-local session
  materialization.
- `src/aiperf/dataset/memory_map_utils.py`: the Python-only implementation
  repaired by origin #59.
