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
const AMDSMI_SUPPORTED_LIBRARY_MAJOR: u32 = 26;

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
type VersionFn = unsafe extern "C" fn(*mut AmdsmiVersion) -> u32;
type SocketHandlesFn = unsafe extern "C" fn(*mut u32, *mut SocketHandle) -> u32;
type ProcessorHandlesFn = unsafe extern "C" fn(SocketHandle, *mut u32, *mut ProcessorHandle) -> u32;
type UuidFn = unsafe extern "C" fn(ProcessorHandle, *mut u32, *mut c_char) -> u32;
type BdfFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiBdf) -> u32;
type AsicInfoFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiAsicInfo) -> u32;
type BoardInfoFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiBoardInfo) -> u32;
type PowerInfoFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiPowerInfo) -> u32;
type EnergyCountFn = unsafe extern "C" fn(ProcessorHandle, *mut u64, *mut f32, *mut u64) -> u32;
type ActivityFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiEngineUsage) -> u32;
type MemoryUsageFn = unsafe extern "C" fn(ProcessorHandle, u32, *mut u64) -> u32;
type TemperatureFn = unsafe extern "C" fn(ProcessorHandle, u32, u32, *mut i64) -> u32;
type EccCountFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiErrorCount) -> u32;
type GpuMetricsFn = unsafe extern "C" fn(ProcessorHandle, *mut AmdsmiGpuMetrics) -> u32;

#[repr(C)]
struct AmdsmiVersion {
    major: u32,
    minor: u32,
    release: u32,
    build: *const c_char,
}

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
struct AmdsmiBoardInfo {
    model_number: [c_char; 256],
    product_serial: [c_char; 256],
    fru_id: [c_char; 256],
    product_name: [c_char; 256],
    manufacturer_name: [c_char; 256],
    reserved: [u64; 64],
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
        validate_library_abi(&library)?;
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

fn validate_library_abi(library: &Library) -> Result<(), GpuTelemetryError> {
    let version =
        unsafe { library.get::<VersionFn>(b"amdsmi_get_lib_version\0") }.map_err(|error| {
            GpuTelemetryError::Worker(format!("loading AMD SMI version query: {error}"))
        })?;
    let mut reported = std::mem::MaybeUninit::<AmdsmiVersion>::zeroed();
    status(
        unsafe { version(reported.as_mut_ptr()) },
        "querying AMD SMI library version",
    )?;
    // The hand-written metrics declaration is verified against ROCm's 26.x ABI.
    let reported = unsafe { reported.assume_init() };
    if reported.major != AMDSMI_SUPPORTED_LIBRARY_MAJOR {
        return Err(GpuTelemetryError::Worker(format!(
            "unsupported AMD SMI ABI {}.{}.{}, expected major {AMDSMI_SUPPORTED_LIBRARY_MAJOR}",
            reported.major, reported.minor, reported.release
        )));
    }
    Ok(())
}

impl VendorWorker for AmdSmiWorker {
    fn initialize(&mut self) -> Result<(), GpuTelemetryError> {
        self.initialize_devices()
    }

    fn scrape(&mut self, timestamp_ns: i64) -> Result<Vec<GpuTelemetryRecord>, GpuTelemetryError> {
        let mut records = Vec::with_capacity(self.devices.len());
        for (index, &device_address) in self.devices.iter().enumerate() {
            let device = device_address as ProcessorHandle;
            if let Some(record) =
                record_from_observation(timestamp_ns, observe_device(&self.library, device, index))
            {
                records.push(record);
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

struct AmdSmiDeviceObservation {
    index: usize,
    gpu_uuid: Option<String>,
    gpu_model_name: Option<String>,
    bdf: Option<AmdsmiBdf>,
    power_candidates: Option<[u64; 3]>,
    activity: Option<(u32, u32, u32)>,
    memory_used_bytes: Option<u64>,
    temperature: Option<i64>,
    ecc_uncorrectable: Option<u64>,
    throttle_status: Option<(u32, u64)>,
    energy: Option<(u64, f32)>,
}

fn observe_device(
    library: &Library,
    device: ProcessorHandle,
    index: usize,
) -> AmdSmiDeviceObservation {
    let mut uuid = [0_i8; AMDSMI_UUID_LENGTH as usize];
    let mut uuid_length = AMDSMI_UUID_LENGTH;
    let gpu_uuid = unsafe { library.get::<UuidFn>(b"amdsmi_get_gpu_device_uuid\0") }
        .ok()
        .filter(|function| unsafe { function(device, &mut uuid_length, uuid.as_mut_ptr()) } == AMDSMI_SUCCESS)
        .and_then(|_| c_string(&uuid));
    let mut board: AmdsmiBoardInfo = unsafe { std::mem::zeroed() };
    let gpu_model_name = unsafe { library.get::<BoardInfoFn>(b"amdsmi_get_gpu_board_info\0") }
        .ok()
        .filter(|function| unsafe { function(device, &mut board) } == AMDSMI_SUCCESS)
        .and_then(|_| c_string(&board.product_name))
        .or_else(|| {
            let mut asic: AmdsmiAsicInfo = unsafe { std::mem::zeroed() };
            unsafe { library.get::<AsicInfoFn>(b"amdsmi_get_gpu_asic_info\0") }
                .ok()
                .filter(|function| unsafe { function(device, &mut asic) } == AMDSMI_SUCCESS)
                .and_then(|_| c_string(&asic.market_name))
        });
    let mut bdf = AmdsmiBdf { as_uint: 0 };
    let bdf = unsafe { library.get::<BdfFn>(b"amdsmi_get_gpu_device_bdf\0") }
        .ok()
        .filter(|function| unsafe { function(device, &mut bdf) } == AMDSMI_SUCCESS)
        .map(|_| bdf);
    let mut power: AmdsmiPowerInfo = unsafe { std::mem::zeroed() };
    let power_candidates = unsafe { library.get::<PowerInfoFn>(b"amdsmi_get_power_info\0") }
        .ok()
        .filter(|function| unsafe { function(device, &mut power) } == AMDSMI_SUCCESS)
        .map(|_| {
            [
                power.socket_power,
                power.current_socket_power as u64,
                power.average_socket_power as u64,
            ]
        });
    let mut activity: AmdsmiEngineUsage = unsafe { std::mem::zeroed() };
    let activity = unsafe { library.get::<ActivityFn>(b"amdsmi_get_gpu_activity\0") }
        .ok()
        .filter(|function| unsafe { function(device, &mut activity) } == AMDSMI_SUCCESS)
        .map(|_| {
            (
                activity.gfx_activity,
                activity.umc_activity,
                activity.mm_activity,
            )
        });
    let mut memory_used_bytes = 0_u64;
    let memory_used_bytes = unsafe { library.get::<MemoryUsageFn>(b"amdsmi_get_gpu_memory_usage\0") }
        .ok()
        .filter(|function| unsafe { function(device, 0, &mut memory_used_bytes) } == AMDSMI_SUCCESS)
        .map(|_| memory_used_bytes);
    let mut temperature = 0_i64;
    let temperature = unsafe { library.get::<TemperatureFn>(b"amdsmi_get_temp_metric\0") }
        .ok()
        .filter(|function| unsafe { function(device, 1, 0, &mut temperature) } == AMDSMI_SUCCESS)
        .map(|_| temperature);
    let mut ecc: AmdsmiErrorCount = unsafe { std::mem::zeroed() };
    let ecc_uncorrectable =
        unsafe { library.get::<EccCountFn>(b"amdsmi_get_gpu_total_ecc_count\0") }
            .ok()
            .filter(|function| unsafe { function(device, &mut ecc) } == AMDSMI_SUCCESS)
            .map(|_| ecc.uncorrectable_count);
    let mut gpu_metrics: AmdsmiGpuMetrics = unsafe { std::mem::zeroed() };
    let throttle_status = unsafe { library.get::<GpuMetricsFn>(b"amdsmi_get_gpu_metrics_info\0") }
        .ok()
        .filter(|function| unsafe { function(device, &mut gpu_metrics) } == AMDSMI_SUCCESS)
        .map(|_| {
            (
                gpu_metrics.throttle_status,
                gpu_metrics.independent_throttle_status,
            )
        });
    let mut energy = 0_u64;
    let mut resolution = 0_f32;
    let mut timestamp = 0_u64;
    let energy = unsafe { library.get::<EnergyCountFn>(b"amdsmi_get_energy_count\0") }
        .ok()
        .filter(|function| unsafe { function(device, &mut energy, &mut resolution, &mut timestamp) } == AMDSMI_SUCCESS)
        .map(|_| (energy, resolution));
    AmdSmiDeviceObservation {
        index,
        gpu_uuid,
        gpu_model_name,
        bdf,
        power_candidates,
        activity,
        memory_used_bytes,
        temperature,
        ecc_uncorrectable,
        throttle_status,
        energy,
    }
}

fn record_from_observation(
    timestamp_ns: i64,
    observation: AmdSmiDeviceObservation,
) -> Option<GpuTelemetryRecord> {
    let mut metrics = BTreeMap::new();
    insert_finite(
        &mut metrics,
        "amd_power",
        observation.power_candidates.and_then(|values| {
            values
                .into_iter()
                .find(|value| *value > 0 && *value < u32::MAX as u64)
                .map(|value| value as f64)
        }),
    );
    if let Some((gfx, umc, mm)) = observation.activity {
        insert_finite(
            &mut metrics,
            "amd_gfx_activity",
            valid_u32(gfx).map(f64::from),
        );
        insert_finite(
            &mut metrics,
            "amd_umc_activity",
            valid_u32(umc).map(f64::from),
        );
        insert_finite(
            &mut metrics,
            "amd_mm_activity",
            valid_u32(mm).map(f64::from),
        );
    }
    insert_finite(
        &mut metrics,
        "amd_memory_used",
        observation
            .memory_used_bytes
            .filter(|value| *value != u64::MAX)
            .map(bytes_to_gigabytes),
    );
    insert_finite(
        &mut metrics,
        "amd_temperature",
        observation
            .temperature
            .filter(|value| *value != i64::MAX)
            .map(|value| {
                let value = value as f64;
                if value > 200.0 { value * 1e-3 } else { value }
            }),
    );
    insert_finite(
        &mut metrics,
        "amd_ecc_uncorrectable",
        observation
            .ecc_uncorrectable
            .filter(|value| *value != u64::MAX)
            .map(|value| value as f64),
    );
    insert_finite(
        &mut metrics,
        "amd_throttle_status",
        observation
            .throttle_status
            .and_then(|(status, independent)| throttle_status_value(status, independent)),
    );
    insert_finite(
        &mut metrics,
        "amd_energy_consumption",
        observation
            .energy
            .map(|(count, resolution)| energy_count_to_megajoules(count, resolution)),
    );
    (!metrics.is_empty()).then_some(GpuTelemetryRecord {
        timestamp_ns,
        endpoint_url: AMDSMI_ENDPOINT_URL.to_string(),
        metadata: metadata_from_parts(
            observation.index,
            observation
                .gpu_uuid
                .unwrap_or_else(|| format!("GPU-unknown-{}", observation.index)),
            observation
                .gpu_model_name
                .unwrap_or_else(|| "Unknown GPU".to_string()),
            observation.bdf.map(pci_bus_id),
        ),
        metrics,
    })
}

fn metadata_from_parts(
    index: usize,
    gpu_uuid: String,
    gpu_model_name: String,
    pci_bus_id: Option<String>,
) -> GpuMetadata {
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

fn pci_bus_id(bdf: AmdsmiBdf) -> String {
    format!(
        "{:04x}:{:02x}:{:02x}.{}",
        bdf.as_uint >> 16,
        (bdf.as_uint >> 8) & 0xff,
        (bdf.as_uint >> 3) & 0x1f,
        bdf.as_uint & 0x7,
    )
}

fn valid_u32(value: u32) -> Option<u32> {
    (value != u32::MAX).then_some(value)
}

fn bytes_to_gigabytes(value: u64) -> f64 {
    value as f64 * 1e-9
}

fn energy_count_to_megajoules(count: u64, resolution: f32) -> f64 {
    count as f64 * resolution as f64 * 1e-12
}

fn is_throttled(status: u32, independent_status: u64) -> bool {
    status != 0 || independent_status != 0
}

fn throttle_status_value(status: u32, independent_status: u64) -> Option<f64> {
    let status = (status != u32::MAX).then_some(status);
    let independent_status = (independent_status != u64::MAX).then_some(independent_status);
    match (status, independent_status) {
        (None, None) => None,
        (status, independent_status) => Some(f64::from(is_throttled(
            status.unwrap_or(0),
            independent_status.unwrap_or(0),
        ))),
    }
}

fn insert_finite(metrics: &mut BTreeMap<String, f64>, name: &str, value: Option<f64>) {
    if let Some(value) = value.filter(|value| value.is_finite() && *value < u32::MAX as f64) {
        metrics.insert(name.to_string(), value);
    }
}

fn c_string(value: &[c_char]) -> Option<String> {
    let bytes = value.iter().map(|byte| *byte as u8).collect::<Vec<_>>();
    let end = bytes.iter().position(|byte| *byte == 0)?;
    std::str::from_utf8(&bytes[..end])
        .ok()
        .map(ToOwned::to_owned)
}

fn status(result: u32, operation: &str) -> Result<(), GpuTelemetryError> {
    (result == AMDSMI_SUCCESS).then_some(()).ok_or_else(|| {
        GpuTelemetryError::Worker(format!("AMD SMI {operation} failed with status {result}"))
    })
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
    fn origin_main_fixture_assembles_complete_native_amdsmi_record() {
        let fixture = serde_json::from_str::<Vec<FixtureRecord>>(include_str!(
            "../../tests/data/gpu_telemetry/amdsmi_origin_main.json"
        ))
        .unwrap()
        .pop()
        .unwrap();
        let actual = record_from_observation(
            fixture.timestamp_ns,
            AmdSmiDeviceObservation {
                index: 0,
                gpu_uuid: Some("GPU-amd".to_string()),
                gpu_model_name: Some("MI300X".to_string()),
                bdf: Some(AmdsmiBdf {
                    as_uint: (0x41_u64 << 8),
                }),
                power_candidates: Some([300, 0, 0]),
                activity: Some((70, 30, u32::MAX)),
                memory_used_bytes: Some(16_000_000_000),
                temperature: Some(72),
                ecc_uncorrectable: Some(3),
                throttle_status: Some((0, 1)),
                energy: Some((2_000_000, 2.0)),
            },
        );
        assert_eq!(actual, Some(fixture.into_record()));
    }

    #[test]
    fn throttle_uses_each_supported_status_independently() {
        let primary_unsupported = record_from_observation(
            1,
            AmdSmiDeviceObservation {
                index: 0,
                gpu_uuid: None,
                gpu_model_name: None,
                bdf: None,
                power_candidates: None,
                activity: None,
                memory_used_bytes: None,
                temperature: None,
                ecc_uncorrectable: None,
                throttle_status: Some((u32::MAX, 1)),
                energy: None,
            },
        )
        .unwrap();
        assert_eq!(
            primary_unsupported.metrics.get("amd_throttle_status"),
            Some(&1.0)
        );

        let independent_unsupported = record_from_observation(
            1,
            AmdSmiDeviceObservation {
                index: 0,
                gpu_uuid: None,
                gpu_model_name: None,
                bdf: None,
                power_candidates: None,
                activity: None,
                memory_used_bytes: None,
                temperature: None,
                ecc_uncorrectable: None,
                throttle_status: Some((0, u64::MAX)),
                energy: None,
            },
        )
        .unwrap();
        assert_eq!(
            independent_unsupported.metrics.get("amd_throttle_status"),
            Some(&0.0)
        );
    }

    #[test]
    fn formats_bdf_from_the_documented_bit_layout() {
        let bdf = AmdsmiBdf {
            as_uint: (0x1234_u64 << 16) | (0xab_u64 << 8) | (0x1c_u64 << 3) | 0x5,
        };
        assert_eq!(pci_bus_id(bdf), "1234:ab:1c.5");
    }

    #[test]
    fn independent_throttle_signal_is_authoritative() {
        assert!(is_throttled(0, 1));
        assert!(is_throttled(1, 0));
        assert!(!is_throttled(0, 0));
    }
}
