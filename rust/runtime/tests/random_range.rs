// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native random-range policy integration tests.

use aiperf_runtime::dataset::{RandomCorpusStyle, RandomRangePlan, RandomRangeRatio};

#[test]
fn style_bounds_and_validation_match_reference_contracts() {
    let vllm = RandomRangePlan::new(
        RandomCorpusStyle::Vllm,
        100,
        20,
        RandomRangeRatio::new(0.3, 0.5).unwrap(),
        2,
    )
    .unwrap();
    assert_eq!(vllm.input_bounds(), (68, 128));
    assert_eq!(vllm.output_bounds(), (10, 30));

    let sglang = RandomRangePlan::new(
        RandomCorpusStyle::Sglang,
        100,
        20,
        RandomRangeRatio::same(0.5).unwrap(),
        2,
    )
    .unwrap();
    assert_eq!(sglang.input_bounds(), (50, 100));
    assert_eq!(sglang.output_bounds(), (10, 20));
    assert_eq!(sglang.adjust_input(50), 48);

    assert!(
        RandomRangeRatio::same(1.0)
            .unwrap()
            .validate(RandomCorpusStyle::Vllm)
            .is_err()
    );
    assert!(
        RandomRangeRatio::new(0.2, 0.3)
            .unwrap()
            .validate(RandomCorpusStyle::Sglang)
            .is_err()
    );
    assert!(
        RandomRangeRatio::same(1.0)
            .unwrap()
            .validate(RandomCorpusStyle::Sglang)
            .is_ok()
    );
    assert!(RandomRangeRatio::same(f64::NAN).is_err());
}

#[test]
fn preseed_matches_numpy_draw_order_for_each_style() {
    let vllm = RandomRangePlan::new(
        RandomCorpusStyle::Vllm,
        100,
        20,
        RandomRangeRatio::new(0.3, 0.5).unwrap(),
        0,
    )
    .unwrap()
    .preseed(4, 42, 1_000)
    .unwrap();
    assert_eq!(vllm.inputs(), &[75, 117, 109, 96]);
    assert_eq!(vllm.outputs(), &[19, 28, 11, 24]);
    assert_eq!(vllm.offsets(), &[201, 94, 526, 975]);

    let sglang = RandomRangePlan::new(
        RandomCorpusStyle::Sglang,
        130,
        20,
        RandomRangeRatio::same(70.0 / 130.0).unwrap(),
        0,
    )
    .unwrap()
    .preseed(4, 42, 1_000)
    .unwrap();
    assert_eq!(sglang.inputs(), &[108, 121, 98, 84]);
    assert_eq!(sglang.outputs(), &[20, 17, 14, 16]);
    assert_eq!(sglang.offsets(), &[121, 466, 214, 330]);
}

#[test]
fn sglang_folds_wide_seeds_by_xor_words() {
    let plan = RandomRangePlan::new(
        RandomCorpusStyle::Sglang,
        100,
        20,
        RandomRangeRatio::same(0.5).unwrap(),
        0,
    )
    .unwrap();
    assert_eq!(
        plan.preseed(6, 5, 100).unwrap(),
        plan.preseed(6, (1_u64 << 32) | 4, 100).unwrap()
    );
}
