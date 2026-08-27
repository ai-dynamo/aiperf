// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::streaming::{STREAMING_RUNTIME_COMPILED, STREAMING_S3_COMPILED};

#[cfg(not(feature = "streaming-s3"))]
#[test]
fn lightweight_streaming_inventory_excludes_s3() {
    assert!(STREAMING_RUNTIME_COMPILED);
    assert!(!STREAMING_S3_COMPILED);
}

#[cfg(feature = "streaming-s3")]
#[test]
fn s3_streaming_inventory_includes_runtime() {
    assert!(STREAMING_RUNTIME_COMPILED);
    assert!(STREAMING_S3_COMPILED);
}
