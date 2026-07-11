// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manual live-data acceptance for the official TIGER-Lab/MMLU-Pro corpus.

use std::rc::Rc;

use aiperf::accuracy_dataset::{
    AccuracyDatasetRegistry, HuggingFaceParquetClient, prepare_mmlu_pro_dataset,
};
use aiperf_accuracy::{
    AccuracyBenchmark, AccuracyRegistry, BenchmarkConfig, DatasetSource, MMLU_PRO_DEFAULT_N_SHOTS,
    MmluProBenchmark,
};
use aiperf_clock::{Clock, RealClock};

#[tokio::test]
#[ignore = "downloads the live public MMLU-Pro dataset"]
async fn downloads_and_materializes_official_mmlu_pro() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let clock: Rc<dyn Clock> = RealClock::new();
            let client = HuggingFaceParquetClient::mmlu_pro(clock);
            let cache = std::env::temp_dir()
                .join(format!("aiperf_official_mmlu_pro_{}", std::process::id()));
            let source = prepare_mmlu_pro_dataset(&client, &cache, true)
                .await
                .unwrap();
            assert_eq!(
                source
                    .load_rows(aiperf_accuracy::DatasetSplit::Validation)
                    .unwrap()
                    .len(),
                70
            );
            assert_eq!(
                source
                    .load_rows(aiperf_accuracy::DatasetSplit::Test)
                    .unwrap()
                    .len(),
                12_032
            );
            let problems = MmluProBenchmark
                .load_problems(
                    &source,
                    &BenchmarkConfig {
                        tasks: vec!["chemistry".to_string()],
                        n_shots: MMLU_PRO_DEFAULT_N_SHOTS,
                        enable_cot: true,
                        max_problems: Some(2),
                        max_tokens: None,
                    },
                )
                .unwrap();
            assert_eq!(problems.len(), 2);
            assert!(problems[0].messages[0].content.contains("about chemistry"));
            assert_eq!(problems[0].ground_truth.len(), 1);
            std::fs::remove_dir_all(cache).unwrap();
        })
        .await;
}

#[tokio::test]
#[ignore = "downloads a live public Hugging Face auto-converted Parquet split"]
async fn downloads_and_materializes_official_aime24_through_registry() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let clock: Rc<dyn Clock> = RealClock::new();
            let cache = std::env::temp_dir().join(format!(
                "aiperf_official_accuracy_catalog_{}",
                std::process::id()
            ));
            let source = AccuracyDatasetRegistry::builtin()
                .prepare("aime24", clock, &cache, true)
                .await
                .unwrap();
            assert_eq!(
                source
                    .load_rows(aiperf_accuracy::DatasetSplit::Train)
                    .unwrap()
                    .len(),
                30
            );
            let registered = AccuracyRegistry::builtin().benchmark("aime24").unwrap();
            let problems = registered
                .load_problems(
                    &source,
                    &BenchmarkConfig {
                        tasks: vec![],
                        n_shots: registered.metadata.default_n_shots,
                        enable_cot: registered.metadata.default_enable_cot,
                        max_problems: Some(2),
                        max_tokens: None,
                    },
                    None,
                )
                .unwrap();
            assert_eq!(problems.len(), 2);
            assert_eq!(problems[0].messages.len(), 1);
            assert_eq!(problems[0].generation.max_tokens, 32_768);
            std::fs::remove_dir_all(cache).unwrap();
        })
        .await;
}
