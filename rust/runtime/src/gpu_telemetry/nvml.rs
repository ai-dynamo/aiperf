// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native NVML telemetry collected on the dedicated vendor worker thread.

use std::collections::BTreeMap;
use std::rc::Rc;

use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{PerformancePolicy, TemperatureSensor};
use nvml_wrapper_sys::bindings::nvmlReturn_enum_NVML_SUCCESS;

use crate::clock::Clock;
use crate::gpu_telemetry::model::{GpuMetadata, GpuTelemetryRecord, NVIDIA_GPU_TELEMETRY_PLATFORM};
use crate::gpu_telemetry::source::GpuTelemetryError;
use crate::gpu_telemetry::vendor_worker::{VendorWorker, VendorWorkerSource};

const NVML_ENDPOINT_URL: &str = "pynvml://localhost";

/// Native local NVIDIA telemetry source backed by NVML's runtime-loaded library.
pub(crate) struct NvmlTelemetrySource {
    worker: VendorWorkerSource,
}

impl NvmlTelemetrySource {
    /// Initializes NVML on its dedicated vendor worker thread.
    pub(crate) async fn spawn(clock: Rc<dyn Clock>) -> Result<Self, GpuTelemetryError> {
        Ok(Self {
            worker: VendorWorkerSource::spawn(clock, NVML_ENDPOINT_URL, || {
                Ok(Box::new(NvmlWorker { nvml: None }))
            })
            .await?,
        })
    }
}

#[async_trait::async_trait(?Send)]
impl crate::gpu_telemetry::source::GpuTelemetrySource for NvmlTelemetrySource {
    fn endpoint_url(&self) -> &str {
        self.worker.endpoint_url()
    }

    async fn scrape(
        &self,
        mode: crate::gpu_telemetry::source::GpuScrapeMode,
    ) -> Result<Option<crate::gpu_telemetry::model::GpuScrape>, GpuTelemetryError> {
        self.worker.scrape(mode).await
    }

    async fn shutdown(&self) -> Result<(), GpuTelemetryError> {
        self.worker.shutdown().await
    }
}

struct NvmlWorker {
    nvml: Option<Nvml>,
}

impl VendorWorker for NvmlWorker {
    fn initialize(&mut self) -> Result<(), GpuTelemetryError> {
        let nvml = Nvml::init().map_err(nvml_error)?;
        let device_count = nvml.device_count().map_err(nvml_error)?;
        self.nvml = Some(nvml);
        if device_count == 0 {
            return Err(GpuTelemetryError::Worker(
                "NVML initialized but no NVIDIA devices are available".to_string(),
            ));
        }
        Ok(())
    }

    fn scrape(&mut self, timestamp_ns: i64) -> Result<Vec<GpuTelemetryRecord>, GpuTelemetryError> {
        let nvml = self.nvml.as_ref().ok_or_else(|| {
            GpuTelemetryError::Worker("NVML scrape requested before initialization".to_string())
        })?;
        let count = nvml.device_count().map_err(nvml_error)?;
        let mut records = Vec::with_capacity(count as usize);
        for index in 0..count {
            let device = match nvml.device_by_index(index) {
                Ok(device) => device,
                Err(error) => {
                    tracing::debug!(error = %error, gpu_index = index, component = "gpu_nvml", "skipping inaccessible NVML device");
                    continue;
                }
            };
            let metadata = device_metadata(&device, index);
            let metrics = device_metrics(nvml, &device);
            if !metrics.is_empty() {
                records.push(GpuTelemetryRecord {
                    timestamp_ns,
                    endpoint_url: NVML_ENDPOINT_URL.to_string(),
                    metadata,
                    metrics,
                });
            }
        }
        Ok(records)
    }

    fn shutdown(&mut self) -> Result<(), GpuTelemetryError> {
        if let Some(nvml) = self.nvml.take() {
            nvml.shutdown().map_err(nvml_error)?;
        }
        Ok(())
    }
}

fn device_metadata(device: &nvml_wrapper::Device<'_>, index: u32) -> GpuMetadata {
    let pci_bus_id = device.pci_info().ok().map(|pci| pci.bus_id);
    GpuMetadata {
        gpu_index: index.min(i32::MAX as u32) as i32,
        gpu_uuid: device.uuid().unwrap_or_else(|_| format!("GPU-{index}")),
        gpu_model_name: device.name().unwrap_or_else(|_| "Unknown".to_string()),
        pci_bus_id,
        device: None,
        hostname: None,
        namespace: None,
        pod_name: None,
        platform: NVIDIA_GPU_TELEMETRY_PLATFORM.to_string(),
    }
}

fn device_metrics(nvml: &Nvml, device: &nvml_wrapper::Device<'_>) -> BTreeMap<String, f64> {
    let mut metrics = BTreeMap::new();
    insert_result(
        &mut metrics,
        "nvidia_power_usage",
        device.power_usage().map(|value| value as f64 * 1e-3),
    );
    insert_result(
        &mut metrics,
        "nvidia_energy_consumption",
        device
            .total_energy_consumption()
            .map(|value| value as f64 * 1e-9),
    );
    insert_result(
        &mut metrics,
        "nvidia_gpu_utilization",
        device.utilization_rates().map(|value| value.gpu as f64),
    );
    insert_result(
        &mut metrics,
        "nvidia_memory_utilization",
        device.utilization_rates().map(|value| value.memory as f64),
    );
    insert_result(
        &mut metrics,
        "nvidia_memory_used",
        device.memory_info().map(|value| value.used as f64 * 1e-9),
    );
    insert_result(
        &mut metrics,
        "nvidia_temperature",
        device.temperature(TemperatureSensor::Gpu).map(f64::from),
    );
    insert_result(
        &mut metrics,
        "nvidia_encoder_utilization",
        device
            .encoder_utilization()
            .map(|value| value.utilization as f64),
    );
    insert_result(
        &mut metrics,
        "nvidia_decoder_utilization",
        device
            .decoder_utilization()
            .map(|value| value.utilization as f64),
    );
    if let Ok(samples) = device.process_utilization_stats(None) {
        let sm_utilization = samples
            .iter()
            .map(|sample| sample.sm_util as f64)
            .sum::<f64>()
            .min(100.0);
        metrics.insert("nvidia_sm_utilization".to_string(), sm_utilization);
    }
    if let Some(utilization) = jpg_utilization(nvml, device) {
        metrics.insert("nvidia_jpg_utilization".to_string(), utilization);
    }
    insert_result(
        &mut metrics,
        "nvidia_power_violation",
        device
            .violation_status(PerformancePolicy::Power)
            .map(|value| value.violation_time as f64 * 1e-3),
    );
    metrics
}

fn jpg_utilization(nvml: &Nvml, device: &nvml_wrapper::Device<'_>) -> Option<f64> {
    let symbol = nvml.lib().nvmlDeviceGetJpgUtilization.as_ref().ok()?;
    let mut utilization = 0_u32;
    let mut sampling_period_us = 0_u32;
    // SAFETY: `device` was resolved from `nvml`, the dynamically loaded symbol
    // has the exact `nvmlDeviceGetJpgUtilization` signature, and both output
    // pointers reference initialized writable local storage for this call.
    let status = unsafe { symbol(device.handle(), &mut utilization, &mut sampling_period_us) };
    (status == nvmlReturn_enum_NVML_SUCCESS).then_some(utilization as f64)
}

fn insert_result(
    metrics: &mut BTreeMap<String, f64>,
    name: &str,
    result: Result<f64, nvml_wrapper::error::NvmlError>,
) {
    if let Ok(value) = result {
        if value.is_finite() {
            metrics.insert(name.to_string(), value);
        }
    }
}

fn nvml_error(error: nvml_wrapper::error::NvmlError) -> GpuTelemetryError {
    GpuTelemetryError::Worker(format!("NVML: {error}"))
}
