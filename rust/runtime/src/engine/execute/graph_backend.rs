// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online graph phase backend factory wiring.

use super::*;

pub(crate) struct OnlineGraphPhaseBackendFactory<'a> {
    pub(crate) placement: &'a dyn GraphPlacementFactory,
    pub(crate) worker_count: usize,
    /// The run's injected clock, handed to the placement so a single-reactor
    /// (virtual) run drives its backend on the `SimClock` rather than a
    /// reconstructed `RealClock`.
    pub(crate) clock: Rc<dyn Clock>,
    pub(crate) real_clock_anchor: RealClockAnchor,
    pub(crate) run_origin_ns: i64,
    pub(crate) model: String,
    pub(crate) default_max_tokens: usize,
    pub(crate) endpoint_runtime_factory: Arc<dyn GraphEndpointRuntimeFactory>,
    pub(crate) segments: Arc<dyn crate::dataset::SegmentStore>,
    pub(crate) metrics: MetricsConfig,
    pub(crate) raw_enabled: bool,
    pub(crate) on_failure: OnFailure,
    pub(crate) cache_bust: Option<crate::engine::graph_execution::GraphCacheBust>,
}

impl GraphPhaseBackendFactory for OnlineGraphPhaseBackendFactory<'_> {
    fn prepare_backend(
        &self,
        config: GraphPhaseBackendConfig,
    ) -> Result<PreparedGraphPhaseBackend> {
        let worker_factory = Arc::new(GraphBackendFactory::new(GraphBackendFactoryConfig {
            real_clock_anchor: self.real_clock_anchor,
            run_origin_ns: self.run_origin_ns,
            model: self.model.clone(),
            default_max_tokens: self.default_max_tokens,
            endpoint_runtime_factory: self.endpoint_runtime_factory.clone(),
            segments: self.segments.clone(),
            metrics: self.metrics.clone(),
            phase: config.metrics_phase,
            prefill_concurrency: config.prefill_concurrency,
            cancellation: config.cancellation,
            raw_enabled: self.raw_enabled,
            events: config.events,
            on_failure: self.on_failure,
            cache_bust: self.cache_bust.clone(),
        }));
        let requires_node_records = self.placement.requires_node_records();
        let placement =
            self.placement
                .build(self.worker_count, worker_factory, self.clock.clone())?;
        Ok(PreparedGraphPhaseBackend {
            placement,
            requires_node_records,
        })
    }
}
