// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SPMC add-only broadcast of dataset chunks.
//!
//! The controller broadcasts each generated chunk once. Late cells replay prior
//! chunks, and every cell indexes only its owned requests by stable `request_id`,
//! independent of chunk arrival order.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::broadcast::{Broadcast, Subscription};

/// One dataset request with its stable global `request_id` (the dispatch position a
/// single-cell run would assign). `R` is the dataset payload (a compiled request / turn).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetRequest<R> {
    /// Stable global position used by the local index and issue commands, never
    /// the arrival order.
    pub request_id: u64,
    /// The compiled request payload.
    pub payload: R,
}

/// A batch of dataset requests the producer adds as one unit. Order-insensitive across
/// chunks and within (each request carries its own `request_id`), so the consumer never
/// depends on chunk or arrival order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetChunk<R> {
    /// Monotonic chunk sequence used for diagnostics and shard availability.
    pub chunk_id: u64,
    /// The requests in this chunk.
    pub requests: Vec<DatasetRequest<R>>,
}

/// Producer side (the controller): generates the dataset once and broadcasts it chunk by
/// chunk. Cheap to clone (shares one broadcast).
#[derive(Clone)]
pub struct DatasetPublisher<R: Clone> {
    broadcast: Broadcast<DatasetChunk<R>>,
    next_chunk: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl<R: Clone> Default for DatasetPublisher<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Clone> DatasetPublisher<R> {
    /// A fresh, empty dataset publisher.
    pub fn new() -> Self {
        Self {
            broadcast: Broadcast::new(),
            next_chunk: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Add one chunk and return its assigned `chunk_id`.
    pub fn add(&self, requests: Vec<DatasetRequest<R>>) -> u64 {
        let chunk_id = self
            .next_chunk
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.broadcast.add(DatasetChunk { chunk_id, requests });
        chunk_id
    }

    /// Seal the dataset: no more chunks. A cell attaching after this still replays every
    /// chunk plus the terminal, so its index is complete.
    pub fn finalize(&self) {
        self.broadcast.finalize();
    }

    /// The number of chunks added so far.
    pub fn chunk_count(&self) -> u64 {
        self.next_chunk.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// A raw broadcast subscription of chunks (for the velo distribution layer). In-process
    /// consumers use [`DatasetIndex::build_owned`].
    pub fn attach_raw(&self) -> Subscription<DatasetChunk<R>> {
        self.broadcast.attach()
    }
}

/// Consumer side (a cell): the local index of the requests this cell owns, keyed by
/// `request_id`. Draining through `Finalized` retains only owned requests.
pub struct DatasetIndex<R> {
    owned: HashMap<u64, R>,
}

impl<R: Clone> DatasetIndex<R> {
    /// Build an index from already-filtered owned entries in a transport adapter.
    pub(crate) fn from_owned(owned: HashMap<u64, R>) -> Self {
        Self { owned }
    }

    /// Drain the dataset broadcast subscription to finalize, indexing only the requests
    /// this cell owns (`owns(request_id)` true). Each chunk is filtered as it streams and
    /// non-owned requests are dropped immediately, so the index's peak RAM is O(owned) —
    /// the cell's ~1/N shard — plus one in-flight chunk, not O(dataset).
    /// Requires the producer to `finalize` (else the live tail blocks forever).
    pub async fn build_owned(
        sub: Subscription<DatasetChunk<R>>,
        owns: impl Fn(u64) -> bool,
    ) -> Self {
        let mut owned = HashMap::new();
        sub.for_each_until_finalized(|chunk| {
            for request in chunk.requests {
                if owns(request.request_id) {
                    owned.insert(request.request_id, request.payload);
                }
            }
        })
        .await;
        Self { owned }
    }

    /// Look up an owned request by stable `request_id`.
    pub fn get(&self, request_id: u64) -> Option<&R> {
        self.owned.get(&request_id)
    }

    /// Whether this cell owns and has indexed the given `request_id`.
    pub fn is_indexed(&self, request_id: u64) -> bool {
        self.owned.contains_key(&request_id)
    }

    /// The number of owned requests indexed.
    pub fn len(&self) -> usize {
        self.owned.len()
    }

    /// Whether this cell owns no requests.
    pub fn is_empty(&self) -> bool {
        self.owned.is_empty()
    }

    /// The owned `request_id`s (sorted), for a cell that dispatches its whole owned slice.
    pub fn owned_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.owned.keys().copied().collect();
        ids.sort_unstable();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-robin ownership predicate: cell `k` of `n` owns `request_id % n == k`.
    fn owns(cell_id: u64, cell_count: u64) -> impl Fn(u64) -> bool {
        move |request_id| request_id % cell_count == cell_id
    }

    /// Publish a dataset of `total` requests in `chunk_size` chunks, request_ids 0..total.
    fn publish(pub_: &DatasetPublisher<String>, total: u64, chunk_size: u64) {
        let mut id = 0;
        while id < total {
            let end = (id + chunk_size).min(total);
            let requests = (id..end)
                .map(|request_id| DatasetRequest {
                    request_id,
                    payload: format!("req-{request_id}"),
                })
                .collect();
            pub_.add(requests);
            id = end;
        }
        pub_.finalize();
    }

    #[tokio::test]
    async fn each_cell_indexes_only_its_owned_shard_keyed_by_request_id() {
        let publisher = DatasetPublisher::<String>::new();
        // Three cells subscribe BEFORE publication (live) — the common case.
        let subs: Vec<_> = (0..3).map(|_| publisher.attach_raw()).collect();
        publish(&publisher, 30, 7);

        let mut total_indexed = 0;
        for (cell_id, sub) in subs.into_iter().enumerate() {
            let index = DatasetIndex::build_owned(sub, owns(cell_id as u64, 3)).await;
            // Cell k owns exactly the request_ids where id % 3 == k.
            let expected: Vec<u64> = (0..30).filter(|id| id % 3 == cell_id as u64).collect();
            assert_eq!(index.owned_ids(), expected, "cell {cell_id} owned set");
            // Index is keyed by request_id (not arrival order) and content-correct.
            for id in &expected {
                assert_eq!(index.get(*id), Some(&format!("req-{id}")));
            }
            // Does NOT hold non-owned requests (O(1/N) RAM).
            assert!(index.get((cell_id as u64 + 1) % 3).is_none() || cell_id == 0);
            total_indexed += index.len();
        }
        // The three owned shards tile the whole dataset with no overlap.
        assert_eq!(total_indexed, 30);
    }

    #[tokio::test]
    async fn a_late_cell_still_indexes_the_full_owned_shard_via_replay() {
        let publisher = DatasetPublisher::<String>::new();
        // Publish everything BEFORE this cell attaches — replay-on-attach must still
        // deliver the whole dataset.
        publish(&publisher, 20, 5);
        let sub = publisher.attach_raw();
        let index = DatasetIndex::build_owned(sub, owns(1, 4)).await;
        let expected: Vec<u64> = (0..20).filter(|id| id % 4 == 1).collect();
        assert_eq!(index.owned_ids(), expected);
        assert_eq!(index.len(), 5);
    }

    #[tokio::test]
    async fn arrival_order_does_not_affect_the_index() {
        // Two publishers add the same requests in DIFFERENT chunk orders; the resulting
        // owned index is identical (keyed by request_id, not arrival).
        let build = |chunks: Vec<Vec<u64>>| async move {
            let publisher = DatasetPublisher::<String>::new();
            let sub = publisher.attach_raw();
            for ids in chunks {
                let requests = ids
                    .into_iter()
                    .map(|request_id| DatasetRequest {
                        request_id,
                        payload: format!("req-{request_id}"),
                    })
                    .collect();
                publisher.add(requests);
            }
            publisher.finalize();
            DatasetIndex::build_owned(sub, owns(0, 2)).await
        };
        let a = build(vec![vec![0, 1, 2], vec![3, 4, 5]]).await;
        let b = build(vec![vec![5, 2], vec![4, 1, 0], vec![3]]).await;
        assert_eq!(a.owned_ids(), b.owned_ids());
        assert_eq!(a.owned_ids(), vec![0, 2, 4]);
    }
}
