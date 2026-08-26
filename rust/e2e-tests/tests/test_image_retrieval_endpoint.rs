// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

#[tokio::test]
async fn test_basic_image_retrieval() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model nvidia/page-elements-v2 \
         --url {} \
         --endpoint-type image_retrieval \
         --endpoint /v1/image/infer \
         --image-width-mean 64 \
         --image-height-mean 64 \
         --request-count 10 \
         --concurrency 2 \
         --workers-max 1 \
         --ui simple",
        h.mock.url
    ));

    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 10);

    let json = r.artifacts.json();
    assert!(json["time_to_first_token"].is_null());
    assert!(!json["request_latency"].is_null());
    assert!(!json["request_throughput"].is_null());
    assert!(!json["image_throughput"].is_null());
    assert!(!json["image_latency"].is_null());
}

#[tokio::test]
async fn yaml_random_pool_honors_cli_image_batch_size() {
    let h = AIPerfHarness::new().await;
    let pool = write_jsonl(
        h.artifact_path(),
        "image_pool.jsonl",
        &[
            serde_json::json!({"image": "data:image/png;base64,iVBORw0KGgo="}),
            serde_json::json!({"image": "data:image/png;base64,iVBORw0KGgoA"}),
        ],
    );
    let config = h.artifact_path().join("image_retrieval_random_pool.yaml");
    std::fs::write(
        &config,
        format!(
            "schemaVersion: \"2.0\"\n\
             benchmark:\n\
            \x20 model: mock-model\n\
            \x20 endpoint: {{type: image_retrieval, url: {}}}\n\
            \x20 dataset: {{type: file, path: {}, format: random_pool}}\n\
            \x20 phases: {{type: concurrency, requests: 2, concurrency: 1}}\n\
            \x20 gpuTelemetry: {{enabled: false}}\n\
            \x20 serverMetrics: {{enabled: false}}\n\
            \x20 runtime: {{ui: none}}\n",
            h.mock.url,
            pool.display(),
        ),
    )
    .expect("write random-pool config");

    let r = h.run(&format!(
        "--config {} --image-batch-size 2",
        config.display()
    ));
    assert!(
        r.success(),
        "image retrieval run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    assert_eq!(r.artifacts.request_count() as u32, 2);
    assert_eq!(
        r.artifacts.json()["total_num_images"]["avg"],
        serde_json::json!(4.0),
        "two requests must each submit the CLI-selected two-image random-pool batch"
    );
}

/// Validates the aggregate and sweep-line image-sample throughput metrics with
/// exact timing/calculation checks against the run's own reported scalars.
#[tokio::test]
async fn test_image_samples_per_second_exact() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model nvidia/page-elements-v2 \
         --url {} \
         --endpoint-type image_retrieval \
         --endpoint /v1/image/infer \
         --image-width-mean 64 \
         --image-height-mean 64 \
         --request-count 20 \
         --concurrency 4 \
         --workers-max 1 \
         --ui simple",
        h.mock.url
    ));

    assert!(r.success());
    let request_count = r.artifacts.request_count();
    assert_eq!(request_count as u32, 20);

    let json = r.artifacts.json();

    // Per-record image count (averaged over requests) and the aggregate sum.
    let num_images_avg = json["num_images"]["avg"]
        .as_f64()
        .expect("num_images avg present");
    assert!(
        num_images_avg >= 1.0,
        "each request submits >=1 image sample"
    );
    let total_num_images = json["total_num_images"]["avg"]
        .as_f64()
        .expect("total_num_images present");

    // Sum == per-request average * request count, exactly (all integer counts).
    let expected_total = num_images_avg * request_count;
    assert!(
        (total_num_images - expected_total).abs() < 1e-9,
        "total_num_images {total_num_images} != num_images_avg {num_images_avg} * requests {request_count}",
    );
    // Every image count is a whole number, so the sum must be integral.
    assert_eq!(
        total_num_images,
        total_num_images.round(),
        "total_num_images must be an integer sample count",
    );

    // Aggregate rate == total samples / benchmark duration (seconds), exactly.
    let benchmark_duration = json["benchmark_duration"]["avg"]
        .as_f64()
        .expect("benchmark_duration present");
    assert!(benchmark_duration > 0.0);
    let samples_per_second = json["image_samples_per_second"]["avg"]
        .as_f64()
        .expect("image_samples_per_second present");
    let expected_rate = total_num_images / benchmark_duration;
    assert!(
        (samples_per_second - expected_rate).abs() / expected_rate < 1e-9,
        "image_samples_per_second {samples_per_second} != total {total_num_images} / duration {benchmark_duration} = {expected_rate}",
    );

    // The sweep-line accumulator variant is present and physically consistent:
    // its duration-weighted average lies at or above the aggregate rate, since
    // the aggregate divides samples over the full run (including any idle tail)
    // while the sweep-line curve weights only the active span.
    let effective = json["effective_image_samples_per_second"]["avg"]
        .as_f64()
        .expect("effective_image_samples_per_second present");
    assert!(
        effective > 0.0,
        "effective_image_samples_per_second must be positive, got {effective}",
    );
    assert!(
        effective + 1e-6 >= expected_rate,
        "sweep-line rate {effective} should be >= aggregate rate {expected_rate}",
    );

    // The active-masked sweep-line sibling is present and at least as high as the
    // effective rate: it averages the sample rate only over intervals where image
    // requests are in flight, so it can only exceed the whole-window average.
    let active = json["active_image_samples_per_second"]["avg"]
        .as_f64()
        .expect("active_image_samples_per_second present");
    assert!(
        active + 1e-6 >= effective,
        "active rate {active} should be >= effective rate {effective}",
    );

    // Per-user sweep-line sibling (design 0006): sample throughput divided by
    // overall concurrency. It is present, positive, and cannot exceed the
    // aggregate effective rate (dividing by concurrency >= 1 only lowers it).
    let per_user = json["effective_image_samples_per_second_per_user"]["avg"]
        .as_f64()
        .expect("effective_image_samples_per_second_per_user present");
    assert!(
        per_user > 0.0,
        "per-user sample rate must be positive, got {per_user}",
    );
    assert!(
        per_user <= effective + 1e-6,
        "per-user rate {per_user} should be <= aggregate effective rate {effective}",
    );
}
