// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Kubernetes contract, credential, and bounded client seams.

pub mod auth;
pub mod bootstrap;
pub mod client;
pub mod command;
pub mod contract;
pub mod error;
pub mod manifest;
pub mod projection;
pub mod render;
pub mod results;
pub mod submission;

#[cfg(test)]
mod tests;
