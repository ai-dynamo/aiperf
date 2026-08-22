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
            if let Some(record) =
                record_from_observation(timestamp_ns, observe_device(nvml, &device, index))
            {
                records.push(record);
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

struct NvmlDeviceObservation {
    index: u32,
    gpu_uuid: Option<String>,
    gpu_model_name: Option<String>,
    pci_bus_id: Option<String>,
    power_millwatts: Option<u32>,
    energy_millijoules: Option<u64>,
    utilization: Option<(u32, u32)>,
    memory_used_bytes: Option<u64>,
    temperature_celsius: Option<u32>,
    encoder_utilization: Option<u32>,
    decoder_utilization: Option<u32>,
    sm_utilization: Option<Vec<u32>>,
    jpg_utilization: Option<u32>,
    power_violation_nanoseconds: Option<u64>,
}

fn observe_device(
    nvml: &Nvml,
    device: &nvml_wrapper::Device<'_>,
    index: u32,
) -> NvmlDeviceObservation {
    NvmlDeviceObservation {
        index,
        gpu_uuid: device.uuid().ok(),
        gpu_model_name: device.name().ok(),
        pci_bus_id: device.pci_info().ok().map(|pci| pci.bus_id),
        power_millwatts: device.power_usage().ok(),
        energy_millijoules: device.total_energy_consumption().ok(),
        utilization: device
            .utilization_rates()
            .ok()
            .map(|value| (value.gpu, value.memory)),
        memory_used_bytes: device.memory_info().ok().map(|value| value.used),
        temperature_celsius: device.temperature(TemperatureSensor::Gpu).ok(),
        encoder_utilization: device
            .encoder_utilization()
            .ok()
            .map(|value| value.utilization),
        decoder_utilization: device
            .decoder_utilization()
            .ok()
            .map(|value| value.utilization),
        sm_utilization: device
            .process_utilization_stats(None)
            .ok()
            .map(|samples| samples.into_iter().map(|sample| sample.sm_util).collect()),
        jpg_utilization: jpg_utilization(nvml, device),
        power_violation_nanoseconds: device
            .violation_status(PerformancePolicy::Power)
            .ok()
            .map(|value| value.violation_time),
    }
}

fn record_from_observation(
    timestamp_ns: i64,
    observation: NvmlDeviceObservation,
) -> Option<GpuTelemetryRecord> {
    let mut metrics = BTreeMap::new();
    insert_finite(
        &mut metrics,
        "nvidia_power_usage",
        observation.power_millwatts.map(milliwatts_to_watts),
    );
    insert_finite(
        &mut metrics,
        "nvidia_energy_consumption",
        observation
            .energy_millijoules
            .map(millijoules_to_megajoules),
    );
    if let Some((gpu, memory)) = observation.utilization {
        insert_finite(&mut metrics, "nvidia_gpu_utilization", Some(gpu as f64));
        insert_finite(
            &mut metrics,
            "nvidia_memory_utilization",
            Some(memory as f64),
        );
    }
    insert_finite(
        &mut metrics,
        "nvidia_memory_used",
        observation.memory_used_bytes.map(bytes_to_gigabytes),
    );
    insert_finite(
        &mut metrics,
        "nvidia_temperature",
        observation.temperature_celsius.map(f64::from),
    );
    insert_finite(
        &mut metrics,
        "nvidia_encoder_utilization",
        observation.encoder_utilization.map(f64::from),
    );
    insert_finite(
        &mut metrics,
        "nvidia_decoder_utilization",
        observation.decoder_utilization.map(f64::from),
    );
    insert_finite(
        &mut metrics,
        "nvidia_sm_utilization",
        observation
            .sm_utilization
            .map(|samples| samples.into_iter().map(f64::from).sum::<f64>().min(100.0)),
    );
    insert_finite(
        &mut metrics,
        "nvidia_jpg_utilization",
        observation.jpg_utilization.map(f64::from),
    );
    insert_finite(
        &mut metrics,
        "nvidia_power_violation",
        observation
            .power_violation_nanoseconds
            .map(nanoseconds_to_microseconds),
    );
    (!metrics.is_empty()).then_some(GpuTelemetryRecord {
        timestamp_ns,
        endpoint_url: NVML_ENDPOINT_URL.to_string(),
        metadata: metadata_from_parts(
            observation.index,
            observation
                .gpu_uuid
                .unwrap_or_else(|| format!("GPU-unknown-{}", observation.index)),
            observation
                .gpu_model_name
                .unwrap_or_else(|| "Unknown GPU".to_string()),
            observation.pci_bus_id,
        ),
        metrics,
    })
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

fn milliwatts_to_watts(value: u32) -> f64 {
    value as f64 * 1e-3
}

fn millijoules_to_megajoules(value: u64) -> f64 {
    value as f64 * 1e-9
}

fn bytes_to_gigabytes(value: u64) -> f64 {
    value as f64 * 1e-9
}

fn nanoseconds_to_microseconds(value: u64) -> f64 {
    value as f64 * 1e-3
}

fn jpg_utilization(nvml: &Nvml, device: &nvml_wrapper::Device<'_>) -> Option<u32> {
    let symbol = nvml.lib().nvmlDeviceGetJpgUtilization.as_ref().ok()?;
    let mut utilization = 0_u32;
    let mut sampling_period_us = 0_u32;
    // SAFETY: `device` was resolved from `nvml`, the dynamically loaded symbol
    // has the exact `nvmlDeviceGetJpgUtilization` signature, and both output
    // pointers reference initialized writable local storage for this call.
    let status = unsafe { symbol(device.handle(), &mut utilization, &mut sampling_period_us) };
    (status == nvmlReturn_enum_NVML_SUCCESS).then_some(utilization)
}

fn insert_finite(metrics: &mut BTreeMap<String, f64>, name: &str, value: Option<f64>) {
    if let Some(value) = value.filter(|value| value.is_finite()) {
        metrics.insert(name.to_string(), value);
    }
}

fn nvml_error(error: nvml_wrapper::error::NvmlError) -> GpuTelemetryError {
    GpuTelemetryError::Worker(format!("NVML: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct FixtureRecord {
        timestamp_ns: i64,
        telemetry_source_url: String,
        gpu_index: i32,
        gpu_uuid: String,
        gpu_model_name: String,
        pci_bus_id: Option<String>,
        device: Option<String>,
        hostname: Option<String>,
        platform: String,
        telemetry_data: BTreeMap<String, f64>,
    }

    impl FixtureRecord {
        fn into_record(self) -> GpuTelemetryRecord {
            GpuTelemetryRecord {
                timestamp_ns: self.timestamp_ns,
                endpoint_url: self.telemetry_source_url,
                metadata: GpuMetadata {
                    gpu_index: self.gpu_index,
                    gpu_uuid: self.gpu_uuid,
                    gpu_model_name: self.gpu_model_name,
                    pci_bus_id: self.pci_bus_id,
                    device: self.device,
                    hostname: self.hostname,
                    namespace: None,
                    pod_name: None,
                    platform: self.platform,
                },
                metrics: self.telemetry_data,
            }
        }
    }

    #[test]
    fn origin_main_fixture_assembles_complete_native_nvml_record() {
        let fixture = serde_json::from_str::<Vec<FixtureRecord>>(include_str!(
            "../../tests/data/gpu_telemetry/nvml_origin_main.json"
        ))
        .unwrap()
        .pop()
        .unwrap();
        let actual = record_from_observation(
            fixture.timestamp_ns,
            NvmlDeviceObservation {
                index: 0,
                gpu_uuid: Some("GPU-nvml".to_string()),
                gpu_model_name: Some("H100".to_string()),
                pci_bus_id: Some("0000:01:00.0".to_string()),
                power_millwatts: Some(250_000),
                energy_millijoules: Some(3_000_000),
                utilization: Some((80, 40)),
                memory_used_bytes: Some(12_000_000_000),
                temperature_celsius: Some(65),
                encoder_utilization: Some(34),
                decoder_utilization: Some(12),
                sm_utilization: Some(vec![25]),
                jpg_utilization: Some(56),
                power_violation_nanoseconds: Some(4_000),
            },
        );
        assert_eq!(actual, Some(fixture.into_record()));
    }
}
