# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Origin #59 findings: mmap conversation cache

Upstream `c9288da6c107c145a310dfe23f64c6ab35328feb` repairs Python's
`MemoryMapDatasetClient`.  Its data mmap had one mutable file cursor, so a
concurrent `seek()` followed by `read()` could return another conversation's
bytes.  The commit replaces that pair with a position-free slice, flattens the
Pydantic offset model for lookup, optionally prefaults data pages, and removes
the async executor hop around each read.

## Ancestry

This exact upstream object is already present in the shared history: target
merge `f1d39ad583f2ed6848b135bf3713240a123a472b` has
`c2889280a66fc85b44e9456fd7020874c73a44fc` as its second parent, and that
upstream #60 object has `c9288da6c1` as its first parent.  `git
merge-base --is-ancestor c9288da6c1 HEAD` succeeds.  A second merge would be a
no-op and would not express new semantic work.

## Native comparison

There is no native mmap dataset backend, mmap cache directory, Python mmap
client-store lifecycle, mutable file cursor, page-prefault setting, or
executor-backed conversation lookup.  Rust constructs a `Dataset` once from
owned conversations, freezes its segment store, and then retains both through
immutable `Arc` storage:

- `runtime/src/dataset/runtime_dataset.rs`: `Dataset` stores
  `Arc<[Conversation]>`; `Dataset::get` reads its immutable ID-to-position map
  and returns an immutable slice reference.
- `runtime/src/multiturn.rs`: `NativeDatasetConversationSource` first lowers
  the dataset, then wraps it in `Arc<NativeDataset>` before source creation.
  `WorkerMaterializationRecipe` shares that immutable `Arc` into each worker;
  session state is deliberately worker-local in `RefCell<HashMap<...>>`.

That layout has neither shared read position nor page-faulting mmap access, so
the upstream race and executor latency cannot occur.  Adding a native mmap
cache solely to mirror Python would add an unrequested persistence format,
cache lifecycle, page-fault policy, and synchronization surface without fixing
an existing native behavior.

## Verification and ruling

The existing `dataset::runtime_dataset::tests::insertion_order_lookup_and_metadata_are_preserved`
test exercises ID lookup through the immutable native dataset.  Its scope is
not a substitute for Python's mmap race test: that test requires the absent
shared mutable mmap cursor.  The appropriate port is therefore no production
change and no invented concurrency test; the direct Python test remains owned
by the upstream Python backend.

## Graham review outcome

Self-Graham review covers the closure-only diff.  It finds no production hot
path, synchronization, allocation, error, tracing, or interface change to
review; the records accurately constrain the no-op decision.
