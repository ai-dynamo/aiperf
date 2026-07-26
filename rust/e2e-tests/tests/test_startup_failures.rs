// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::*;

#[tokio::test]
async fn test_invalid_model_name_exits_with_error() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model this-model-does-not-exist-and-will-fail \
             --url {} \
             --request-count 10 \
             --concurrency 2",
            h.mock.url
        ),
        60,
    );
    assert_ne!(
        r.exit_code, 0,
        "Expected non-zero exit code when model/tokenizer fails to load"
    );
}
