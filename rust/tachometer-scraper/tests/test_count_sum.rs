/// Test that _count and _sum metrics are preserved
use tachometer_scraper::parse;

#[test]
fn test_histogram_count_and_sum_preserved() {
    let text = r#"# TYPE test_histogram histogram
test_histogram_bucket{le="0.1"} 10
test_histogram_bucket{le="1.0"} 50
test_histogram_bucket{le="+Inf"} 100
test_histogram_count 100
test_histogram_sum 45.5"#;

    let samples = parse::parse_prometheus_samples(text).unwrap();

    // Should have 5 samples: 3 buckets + count + sum
    assert_eq!(
        samples.len(),
        5,
        "Should parse all samples including _count and _sum"
    );

    // Check for _count
    let count_sample = samples
        .iter()
        .find(|s| matches!(s.metric_type, parse::MetricType::HistogramCount));
    assert!(count_sample.is_some(), "Should have histogram _count");
    assert_eq!(count_sample.unwrap().value, 100.0);

    // Check for _sum
    let sum_sample = samples
        .iter()
        .find(|s| matches!(s.metric_type, parse::MetricType::HistogramSum));
    assert!(sum_sample.is_some(), "Should have histogram _sum");
    assert_eq!(sum_sample.unwrap().value, 45.5);

    // Convert to rows and check that sum/count are attached to all buckets
    let rows = parse::samples_to_rows(samples, "test");

    // Should have 3 bucket rows (no separate _count and _sum rows)
    assert_eq!(rows.len(), 3, "Should have 3 histogram bucket rows");

    // All buckets should have sum and count attached
    for bucket in &rows {
        assert_eq!(
            bucket.histogram_sum,
            Some(45.5),
            "All buckets should have histogram_sum"
        );
        assert_eq!(
            bucket.histogram_count,
            Some(100.0),
            "All buckets should have histogram_count"
        );
    }

    println!("Successfully verified histogram sum and count are attached to all buckets");
}

#[test]
fn test_summary_count_and_sum_preserved() {
    let text = r#"# TYPE go_gc_duration_seconds summary
go_gc_duration_seconds{quantile="0"} 0
go_gc_duration_seconds{quantile="0.5"} 1.5
go_gc_duration_seconds{quantile="1"} 3.0
go_gc_duration_seconds_sum 12.5
go_gc_duration_seconds_count 100"#;

    let samples = parse::parse_prometheus_samples(text).unwrap();

    // Should have 5 samples: 3 quantiles + count + sum
    assert_eq!(
        samples.len(),
        5,
        "Should parse all samples including _count and _sum"
    );

    // Check for _count
    let count_sample = samples
        .iter()
        .find(|s| matches!(s.metric_type, parse::MetricType::SummaryCount));
    assert!(count_sample.is_some(), "Should have summary _count");
    assert_eq!(count_sample.unwrap().value, 100.0);

    // Check for _sum
    let sum_sample = samples
        .iter()
        .find(|s| matches!(s.metric_type, parse::MetricType::SummarySum));
    assert!(sum_sample.is_some(), "Should have summary _sum");
    assert_eq!(sum_sample.unwrap().value, 12.5);

    // Convert to rows and check metric names
    let rows = parse::samples_to_rows(samples, "test");
    let metric_names: Vec<&str> = rows.iter().map(|r| r.metric_name.as_str()).collect();

    assert!(
        metric_names.iter().any(|m| m.contains("_count")),
        "Should have _count in metric names"
    );
    assert!(
        metric_names.iter().any(|m| m.contains("_sum")),
        "Should have _sum in metric names"
    );

    println!("Metric names: {:?}", metric_names);
}
