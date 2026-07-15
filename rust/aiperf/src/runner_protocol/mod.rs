// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runner-protocol layer: the v2 protocol / registry / execution modules
//! relocated out of the `aiperf-runner` crate.
//!
//! This module hosts the ~30k-line v2 execution substrate — protocol envelopes,
//! the frozen transport/workload/pair registries, the execution factories and
//! drivers, dataset/graph input resolution, the coordinator/application
//! composition root, and the ancillary side-channel accumulators — so that the
//! `aiperf-runner` binary is reduced to a thin process shell (`main.rs`, the
//! cellular controller/cell, the control-plane HTTP surface, and signal
//! handling).
//!
//! It is gated behind the `runner-protocol` Cargo feature: only `aiperf-runner`
//! opts in, so `aiperf-mock-server`, `e2e`, and other library consumers pull
//! `aiperf` with default features and never compile this layer or its
//! dependency surface.
//!
//! The relocation tasks `git mv` the runner modules in here leaf-first
//! (protocol → registry → drivers → side-channels) and rewrite their
//! references.

pub mod distribution_identity;
pub mod records;
pub mod redaction;
