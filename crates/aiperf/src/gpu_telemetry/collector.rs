// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-source sequential GPU telemetry collector.
//!
//! Explicit baseline/final collection is driven directly by the native runtime,
//! which calls these methods at phase barriers instead of routing records
//! through services or messages.

use std::rc::Rc;

use crate::gpu_telemetry::accumulator::GpuTelemetryAccumulator;
use crate::gpu_telemetry::model::{GpuBoundarySnapshot, GpuScrape};
use crate::gpu_telemetry::source::{GpuScrapeMode, GpuTelemetryError, GpuTelemetrySource};

/// Drives one injected GPU source into one caller-owned accumulator.
pub struct GpuTelemetryCollector {
    source: Rc<dyn GpuTelemetrySource>,
}

impl GpuTelemetryCollector {
    /// Builds a collector over a DCGM, local, or replay source.
    pub fn new(source: Rc<dyn GpuTelemetrySource>) -> Self {
        Self { source }
    }

    /// Returns the source's credential-free identifier.
    pub fn endpoint_url(&self) -> &str {
        self.source.endpoint_url()
    }

    /// Collects one cadence scrape without borrowing an accumulator across IO.
    ///
    /// Runtimes with interior-mutability-owned stores use this split API so
    /// they can await the source first and then ingest synchronously.
    pub async fn collect_continuous(&self) -> Result<Option<GpuScrape>, GpuTelemetryError> {
        self.source.scrape(GpuScrapeMode::Continuous).await
    }

    /// Collects one mandatory boundary scrape and its exact counter snapshot.
    pub async fn collect_boundary(
        &self,
    ) -> Result<(GpuScrape, GpuBoundarySnapshot), GpuTelemetryError> {
        let scrape = self
            .source
            .scrape(GpuScrapeMode::Boundary)
            .await?
            .expect("boundary sources must not suppress duplicate bodies");
        let boundary = GpuBoundarySnapshot::from_scrape(&scrape);
        Ok((scrape, boundary))
    }

    /// Synchronously appends a completed scrape to caller-owned storage.
    pub fn ingest_scrape(scrape: &GpuScrape, accumulator: &mut GpuTelemetryAccumulator) {
        for record in &scrape.records {
            accumulator.ingest_record(record);
        }
    }

    /// Releases process, device, or transport resources owned by the source.
    pub async fn shutdown(&self) -> Result<(), GpuTelemetryError> {
        self.source.shutdown().await
    }

    /// Performs one cadence scrape and ingests it unless its body is unchanged.
    pub async fn scrape_continuous(
        &self,
        accumulator: &mut GpuTelemetryAccumulator,
    ) -> Result<usize, GpuTelemetryError> {
        let Some(scrape) = self.collect_continuous().await? else {
            return Ok(0);
        };
        let count = scrape.records.len();
        Self::ingest_scrape(&scrape, accumulator);
        Ok(count)
    }

    /// Performs a mandatory boundary scrape, ingests it, and captures counters.
    pub async fn scrape_boundary(
        &self,
        accumulator: &mut GpuTelemetryAccumulator,
    ) -> Result<GpuBoundarySnapshot, GpuTelemetryError> {
        let (scrape, boundary) = self.collect_boundary().await?;
        Self::ingest_scrape(&scrape, accumulator);
        Ok(boundary)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::BTreeMap;

    use async_trait::async_trait;

    use super::*;
    use crate::gpu_telemetry::{GpuMetadata, GpuScrape, GpuTelemetryRecord};

    struct MockSource {
        calls: Cell<usize>,
    }

    #[async_trait(?Send)]
    impl GpuTelemetrySource for MockSource {
        fn endpoint_url(&self) -> &str {
            "mock://gpu"
        }

        async fn scrape(
            &self,
            mode: GpuScrapeMode,
        ) -> Result<Option<GpuScrape>, GpuTelemetryError> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            if call > 0 && mode == GpuScrapeMode::Continuous {
                return Ok(None);
            }
            let record = GpuTelemetryRecord {
                timestamp_ns: call as i64,
                endpoint_url: "mock://gpu".to_string(),
                metadata: GpuMetadata {
                    gpu_index: 0,
                    gpu_uuid: "GPU-0".to_string(),
                    gpu_model_name: "Mock".to_string(),
                    pci_bus_id: None,
                    device: None,
                    hostname: None,
                    namespace: None,
                    pod_name: None,
                },
                metrics: BTreeMap::from([("energy_consumption".to_string(), call as f64)]),
            };
            Ok(Some(GpuScrape {
                timestamp_ns: call as i64,
                endpoint_url: "mock://gpu".to_string(),
                records: vec![record],
            }))
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn boundary_scrapes_are_never_lost_to_cadence_dedup() {
        let collector = GpuTelemetryCollector::new(Rc::new(MockSource {
            calls: Cell::new(0),
        }));
        let mut accumulator = GpuTelemetryAccumulator::new();
        assert_eq!(
            collector.scrape_continuous(&mut accumulator).await.unwrap(),
            1
        );
        assert_eq!(
            collector.scrape_continuous(&mut accumulator).await.unwrap(),
            0
        );
        let boundary = collector.scrape_boundary(&mut accumulator).await.unwrap();
        assert_eq!(boundary.timestamp_ns, 2);
        assert_eq!(accumulator.len(), 2);
    }
}
