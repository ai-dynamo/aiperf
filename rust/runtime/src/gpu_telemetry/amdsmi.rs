// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native AMD SMI telemetry collected on the dedicated vendor worker thread.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_void};
use std::rc::Rc;

use libloading::Library;

use crate::clock::Clock;
use crate::gpu_telemetry::model::{AMD_GPU_TELEMETRY_PLATFORM, GpuMetadata, GpuTelemetryRecord};
use crate::gpu_telemetry::source::{GpuScrapeMode, GpuTelemetryError, GpuTelemetrySource};
use crate::gpu_telemetry::vendor_worker::{VendorWorker, VendorWorkerSource};

const AMDSMI_ENDPOINT_URL: &str = "amdsmi://localhost";
const AMDSMI_SUCCESS: u32 = 0;
const AMDSMI_INIT_AMD_GPUS: u64 = 1 << 1;
const AMDSMI_UUID_LENGTH: u32 = 38;

/// Native local AMD telemetry source backed by the runtime-loaded AMD SMI library.
pub(crate) struct AmdSmiTelemetrySource {
    worker: VendorWorkerSource,
}

impl AmdSmiTelemetrySource {
    /// Initializes AMD SMI on its dedicated vendor worker thread.
    pub(crate) async fn spawn(clock: Rc<dyn Clock>) -> Result<Self, GpuTelemetryError> {
        Ok(Self {
            worker: VendorWorkerSource::spawn(clock, AMDSMI_ENDPOINT_URL, || {
                Ok(Box::new(AmdSmiWorker::new()?))
            })
            .await?,
        })
    }
}

#[async_trait::async_trait(?Send)]
impl GpuTelemetrySource for AmdSmiTelemetrySource {
    fn endpoint_url(&self) -> &str {
        self.worker.endpoint_url()
    }

    async fn scrape(
        &self,
        mode: GpuScrapeMode,
    ) -> Result<Option<crate::gpu_telemetry::GpuScrape>, GpuTelemetryError> {
        self.worker.scrape(mode).await
    }

    async fn shutdown(&self) -> Result<(), GpuTelemetryError> {
        self.worker.shutdown().await
    }
}

type SocketHandle = *mut c_void;
type ProcessorHandle = *mut c_void;
type InitFn = unsafe extern "C" fn(u64) -> u32;
type ShutdownFn = unsafe extern "C" fn() -> u32;
type SocketHandlesFn = unsafe extern "C" fn(*mut u32, *mut SocketHandle) -> u32;
type ProcessorHandlesFn = unsafe extern "C" fn(SocketHandle, *mut u32, *mut ProcessorHandle) -> u32;
type UuidFn = unsafe extern "C" fn(ProcessorHandle, *mut u32, *mut c_char) -> u32;
type BdfFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiBdf) -> u32;
type AsicInfoFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiAsicInfo) -> u32;
type PowerInfoFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiPowerInfo) -> u32;
type EnergyCountFn = unsafe extern "C" fn(ProcessorHandle, *mut u64, *mut f32, *mut u64) -> u32;
type ActivityFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiEngineUsage) -> u32;
type VramUsageFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiVramUsage) -> u32;
type TemperatureFn = unsafe extern "C" fn(ProcessorHandle, u32, u32, *mut i64) -> u32;
type EccCountFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiErrorCount) -> u32;
type GpuMetricsFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiGpuMetrics) -> u32;

#[repr(C)]
#[derive(Clone, Copy)]
struct AmdsmiBdf {
    as_uint: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AmdsmiAsicInfo {
    market_name: [c_char; 256],
    vendor_id: u32,
    vendor_name: [c_char; 256],
    subvendor_id: u32,
    device_id: u64,
    rev_id: u32,
    asic_serial: [c_char; 256],
    oam_id: u32,
    num_of_compute_units: u32,
    target_graphics_version: u64,
    subsystem_id: u32,
    reserved: [u32; 21],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AmdsmiPowerInfo {
    socket_power: u64,
    current_socket_power: u32,
    average_socket_power: u32,
    gfx_voltage: u64,
    soc_voltage: u64,
    mem_voltage: u64,
    power_limit: u32,
    reserved: [u64; 18],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AmdsmiEngineUsage {
    gfx_activity: u32,
    umc_activity: u32,
    mm_activity: u32,
    reserved: [u32; 13],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AmdsmiVramUsage {
    vram_total: u32,
    vram_used: u32,
    reserved: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AmdsmiErrorCount {
    correctable_count: u64,
    uncorrectable_count: u64,
    deferred_count: u64,
    reserved: [u64; 5],
}

#[repr(C, align(8))]
struct AmdsmiGpuMetrics {
    common_header: [u8; 4],
    _prefix: [u8; 64],
    throttle_status: u32,
    _to_independent_throttle_status: [u8; 40],
    independent_throttle_status: u64,
    _remaining: [u8; 4424],
}

struct AmdSmiWorker {
    library: Library,
    // Opaque C handles never leave this dedicated worker thread. Store their
    // address representation so the worker object itself remains movable into it.
    devices: Vec<usize>,
    is_initialized: bool,
}

impl AmdSmiWorker {
    fn new() -> Result<Self, GpuTelemetryError> {
        let library = unsafe { Library::new("libamd_smi.so.26") }
            .or_else(|_| unsafe { Library::new("libamd_smi.so") })
            .map_err(|error| GpuTelemetryError::Worker(format!("loading AMD SMI: {error}")))?;
        Ok(Self {
            library,
            devices: Vec::new(),
            is_initialized: false,
        })
    }

    fn symbol<T>(&self, name: &[u8]) -> Result<libloading::Symbol<'_, T>, GpuTelemetryError> {
        unsafe { self.library.get(name) }
            .map_err(|error| GpuTelemetryError::Worker(format!("loading AMD SMI symbol: {error}")))
    }

    fn initialize_devices(&mut self) -> Result<(), GpuTelemetryError> {
        let init = self.symbol::<InitFn>(b"amdsmi_init\0")?;
        status(
            unsafe { init(AMDSMI_INIT_AMD_GPUS) },
            "initializing AMD SMI",
        )?;
        self.is_initialized = true;
        let sockets = enumerate_sockets(&self.library)?;
        for socket in sockets {
            self.devices.extend(
                enumerate_processors(&self.library, socket)?
                    .into_iter()
                    .map(|handle| handle as usize),
            );
        }
        if self.devices.is_empty() {
            return Err(GpuTelemetryError::Worker(
                "AMD SMI initialized but no AMD GPUs are available".to_string(),
            ));
        }
        Ok(())
    }
}

impl VendorWorker for AmdSmiWorker {
    fn initialize(&mut self) -> Result<(), GpuTelemetryError> {
        self.initialize_devices()
    }

    fn scrape(&mut self, timestamp_ns: i64) -> Result<Vec<GpuTelemetryRecord>, GpuTelemetryError> {
        let mut records = Vec::with_capacity(self.devices.len());
        for (index, &device_address) in self.devices.iter().enumerate() {
            let device = device_address as ProcessorHandle;
            let metadata = metadata(&self.library, device, index);
            let metrics = metrics(&self.library, device);
            if !metrics.is_empty() {
                records.push(GpuTelemetryRecord {
                    timestamp_ns,
                    endpoint_url: AMDSMI_ENDPOINT_URL.to_string(),
                    metadata,
                    metrics,
                });
            }
        }
        Ok(records)
    }

    fn shutdown(&mut self) -> Result<(), GpuTelemetryError> {
        self.devices.clear();
        if self.is_initialized {
            let shutdown = self.symbol::<ShutdownFn>(b"amdsmi_shut_down\0")?;
            status(unsafe { shutdown() }, "shutting down AMD SMI")?;
            self.is_initialized = false;
        }
        Ok(())
    }
}

fn enumerate_sockets(library: &Library) -> Result<Vec<SocketHandle>, GpuTelemetryError> {
    let get = unsafe { library.get::<SocketHandlesFn>(b"amdsmi_get_socket_handles\0") }.map_err(
        |error| GpuTelemetryError::Worker(format!("loading AMD SMI socket enumeration: {error}")),
    )?;
    let mut count = 0;
    status(
        unsafe { get(&mut count, std::ptr::null_mut()) },
        "counting AMD SMI sockets",
    )?;
    let mut sockets = vec![std::ptr::null_mut(); count as usize];
    status(
        unsafe { get(&mut count, sockets.as_mut_ptr()) },
        "enumerating AMD SMI sockets",
    )?;
    sockets.truncate(count as usize);
    Ok(sockets)
}

fn enumerate_processors(
    library: &Library,
    socket: SocketHandle,
) -> Result<Vec<ProcessorHandle>, GpuTelemetryError> {
    let get = unsafe { library.get::<ProcessorHandlesFn>(b"amdsmi_get_processor_handles\0") }
        .map_err(|error| {
            GpuTelemetryError::Worker(format!("loading AMD SMI processor enumeration: {error}"))
        })?;
    let mut count = 0;
    status(
        unsafe { get(socket, &mut count, std::ptr::null_mut()) },
        "counting AMD SMI processors",
    )?;
    let mut processors = vec![std::ptr::null_mut(); count as usize];
    status(
        unsafe { get(socket, &mut count, processors.as_mut_ptr()) },
        "enumerating AMD SMI processors",
    )?;
    processors.truncate(count as usize);
    Ok(processors)
}

fn metadata(library: &Library, device: ProcessorHandle, index: usize) -> GpuMetadata {
    let mut uuid = [0_i8; AMDSMI_UUID_LENGTH as usize];
    let mut uuid_length = AMDSMI_UUID_LENGTH;
    let gpu_uuid = unsafe { library.get::<UuidFn>(b"amdsmi_get_gpu_device_uuid\0") }
        .ok()
        .filter(|function| unsafe { function(device, &mut uuid_length, uuid.as_mut_ptr()) } == AMDSMI_SUCCESS)
        .and_then(|_| c_string(&uuid))
        .unwrap_or_else(|| format!("GPU-{index}"));
    let mut asic: AmdsmiAsicInfo = unsafe { std::mem::zeroed() };
    let gpu_model_name = unsafe { library.get::<AsicInfoFn>(b"amdsmi_get_gpu_asic_info\0") }
        .ok()
        .filter(|function| unsafe { function(device, &mut asic) } == AMDSMI_SUCCESS)
        .and_then(|_| c_string(&asic.market_name))
        .unwrap_or_else(|| "Unknown".to_string());
    let mut bdf = AmdsmiBdf { as_uint: 0 };
    let pci_bus_id = unsafe { library.get::<BdfFn>(b"amdsmi_get_gpu_device_bdf\0") }
        .ok()
        .filter(|function| unsafe { function(device, &mut bdf) } == AMDSMI_SUCCESS)
        .map(|_| {
            format!(
                "{:04x}:{:02x}:{:02x}.{}",
                bdf.as_uint >> 16,
                (bdf.as_uint >> 8) & 0xff,
                (bdf.as_uint >> 3) & 0x1f,
                bdf.as_uint & 0x7,
            )
        });
    GpuMetadata {
        gpu_index: index.min(i32::MAX as usize) as i32,
        gpu_uuid,
        gpu_model_name,
        pci_bus_id,
        device: Some(format!("amd{index}")),
        hostname: Some("localhost".to_string()),
        namespace: None,
        pod_name: None,
        platform: AMD_GPU_TELEMETRY_PLATFORM.to_string(),
    }
}

fn metrics(library: &Library, device: ProcessorHandle) -> BTreeMap<String, f64> {
    let mut metrics = BTreeMap::new();
    let mut power: AmdsmiPowerInfo = unsafe { std::mem::zeroed() };
    if unsafe { library.get::<PowerInfoFn>(b"amdsmi_get_power_info\0") }
        .ok()
        .is_some_and(|function| unsafe { function(device, &mut power) } == AMDSMI_SUCCESS)
    {
        let value = [
            power.socket_power as f64,
            power.current_socket_power as f64,
            power.average_socket_power as f64,
        ]
        .into_iter()
        .find(|value| *value > 0.0 && *value < u32::MAX as f64);
        if let Some(value) = value {
            metrics.insert("amd_power".to_string(), value);
        }
    }
    let mut activity: AmdsmiEngineUsage = unsafe { std::mem::zeroed() };
    if unsafe { library.get::<ActivityFn>(b"amdsmi_get_gpu_activity\0") }
        .ok()
        .is_some_and(|function| unsafe { function(device, &mut activity) } == AMDSMI_SUCCESS)
    {
        insert_finite(
            &mut metrics,
            "amd_gfx_activity",
            activity.gfx_activity as f64,
        );
        insert_finite(
            &mut metrics,
            "amd_umc_activity",
            activity.umc_activity as f64,
        );
        insert_finite(&mut metrics, "amd_mm_activity", activity.mm_activity as f64);
    }
    let mut vram: AmdsmiVramUsage = unsafe { std::mem::zeroed() };
    if unsafe { library.get::<VramUsageFn>(b"amdsmi_get_gpu_vram_usage\0") }
        .ok()
        .is_some_and(|function| unsafe { function(device, &mut vram) } == AMDSMI_SUCCESS)
    {
        insert_finite(
            &mut metrics,
            "amd_memory_used",
            vram.vram_used as f64 * 1.048_576e-3,
        );
    }
    let mut temperature = 0_i64;
    let temperature_result = unsafe { library.get::<TemperatureFn>(b"amdsmi_get_temp_metric\0") }
        .ok()
        .and_then(|function| {
            (unsafe { function(device, 1, 0, &mut temperature) } == AMDSMI_SUCCESS)
                .then_some(temperature)
        });
    if let Some(value) = temperature_result.filter(|value| *value != i64::MAX) {
        let value = value as f64;
        insert_finite(
            &mut metrics,
            "amd_temperature",
            if value > 200.0 { value * 1e-3 } else { value },
        );
    }
    let mut ecc: AmdsmiErrorCount = unsafe { std::mem::zeroed() };
    if unsafe { library.get::<EccCountFn>(b"amdsmi_get_gpu_total_ecc_count\0") }
        .ok()
        .is_some_and(|function| unsafe { function(device, &mut ecc) } == AMDSMI_SUCCESS)
        && ecc.uncorrectable_count != u64::MAX
    {
        insert_finite(
            &mut metrics,
            "amd_ecc_uncorrectable",
            ecc.uncorrectable_count as f64,
        );
    }
    let mut gpu_metrics: AmdsmiGpuMetrics = unsafe { std::mem::zeroed() };
    if unsafe { library.get::<GpuMetricsFn>(b"amdsmi_get_gpu_metrics_info\0") }
        .ok()
        .is_some_and(|function| unsafe { function(device, &mut gpu_metrics) } == AMDSMI_SUCCESS)
        && gpu_metrics.throttle_status != u32::MAX
        && gpu_metrics.independent_throttle_status != u64::MAX
    {
        metrics.insert(
            "amd_throttle_status".to_string(),
            if gpu_metrics.throttle_status != 0 || gpu_metrics.independent_throttle_status != 0 {
                1.0
            } else {
                0.0
            },
        );
    }
    let mut energy = 0_u64;
    let mut resolution = 0_f32;
    let mut timestamp = 0_u64;
    if unsafe { library.get::<EnergyCountFn>(b"amdsmi_get_energy_count\0") }.ok().is_some_and(|function| unsafe { function(device, &mut energy, &mut resolution, &mut timestamp) } == AMDSMI_SUCCESS) {
        insert_finite(&mut metrics, "amd_energy_consumption", energy as f64 * resolution as f64 * 1e-12);
    }
    metrics
}

fn c_string(value: &[c_char]) -> Option<String> {
    let bytes = value.iter().map(|byte| *byte as u8).collect::<Vec<_>>();
    let end = bytes.iter().position(|byte| *byte == 0)?;
    std::str::from_utf8(&bytes[..end])
        .ok()
        .map(ToOwned::to_owned)
}

fn insert_finite(metrics: &mut BTreeMap<String, f64>, name: &str, value: f64) {
    if value.is_finite() && value < u32::MAX as f64 {
        metrics.insert(name.to_string(), value);
    }
}

fn status(result: u32, operation: &str) -> Result<(), GpuTelemetryError> {
    (result == AMDSMI_SUCCESS).then_some(()).ok_or_else(|| {
        GpuTelemetryError::Worker(format!("AMD SMI {operation} failed with status {result}"))
    })
}
