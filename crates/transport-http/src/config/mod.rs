// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client + socket defaults.

pub mod defaults;
pub use defaults::{
    ClientConfig, PreparedTlsClientConfig, PreparedTlsClientConfigError, apply_socket_opts,
};
