// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI plugin composition bootstrap.
//!
//! This module resolves the plugin universe from the lock file before any
//! process effects (sockets, file descriptors, platform resources) are opened.
//! It is the integration point where dynamically-loaded plugins join the
//! application bootstrap.

pub mod compose;
pub mod lock_path;
