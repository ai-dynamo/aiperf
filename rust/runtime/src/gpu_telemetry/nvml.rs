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
    metadata_from_parts(
        index,
        device.uuid().unwrap_or_else(|_| format!("GPU-{index}")),
        device.name().unwrap_or_else(|_| "Unknown".to_string()),
        device.pci_info().ok().map(|pci| pci.bus_id),
    )
}

fn metadata_from_parts(
    index: u32,
    gpu_uuid: String,
    gpu_model_name: String,
    pci_bus_id: Option<String>,
) -> GpuMetadata {
    GpuMetadata {
        gpu_index: index.min(i32::MAX as u32) as i32,
        gpu_uuid,
        gpu_model_name,
        pci_bus_id,
        device: Some(format!("nvidia{index}")),
        hostname: Some("localhost".to_string()),
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
        device.power_usage().map(milliwatts_to_watts),
    );
    insert_result(
        &mut metrics,
        "nvidia_energy_consumption",
        device
            .total_energy_consumption()
            .map(millijoules_to_megajoules),
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
        device
            .memory_info()
            .map(|value| bytes_to_gigabytes(value.used)),
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

fn milliwatts_to_watts(value: u32) -> f64 {
    value as f64 * 1e-3
}

fn millijoules_to_megajoules(value: u64) -> f64 {
    value as f64 * 1e-9
}

fn bytes_to_gigabytes(value: u64) -> f64 {
    value as f64 * 1e-9
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_main_fixture_units_and_identity_use_production_normalizers() {
        let fixture = serde_json::from_str::<serde_json::Value>(include_str!(
            "../../tests/data/gpu_telemetry/nvml_origin_main.json"
        ))
        .unwrap();
        let fixture = fixture.as_array().unwrap().first().unwrap();
        let metrics = &fixture["telemetry_data"];
        assert_eq!(
            milliwatts_to_watts(250_000),
            metrics["nvidia_power_usage"].as_f64().unwrap()
        );
        assert_eq!(
            millijoules_to_megajoules(3_000_000),
            metrics["nvidia_energy_consumption"].as_f64().unwrap()
        );
        assert_eq!(
            bytes_to_gigabytes(12_000_000_000),
            metrics["nvidia_memory_used"].as_f64().unwrap()
        );
        let metadata = metadata_from_parts(
            0,
            "GPU-nvml".to_string(),
            "H100".to_string(),
            Some("0000:01:00.0".to_string()),
        );
        assert_eq!(metadata.gpu_uuid, fixture["gpu_uuid"].as_str().unwrap());
        assert_eq!(
            metadata.gpu_model_name,
            fixture["gpu_model_name"].as_str().unwrap()
        );
        assert_eq!(
            metadata.pci_bus_id.as_deref(),
            fixture["pci_bus_id"].as_str()
        );
        assert_eq!(metadata.device.as_deref(), fixture["device"].as_str());
        assert_eq!(metadata.hostname.as_deref(), fixture["hostname"].as_str());
    }
}
