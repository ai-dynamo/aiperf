// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DCGM Prometheus metrics faker.

use std::fmt::Write;
use std::sync::atomic::{AtomicU32, Ordering};

use aiperf_runtime::rng::RandomGenerator;
use parking_lot::Mutex;

#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub model: &'static str,
    pub memory_gb: u32,
    pub max_power_w: u32,
    pub idle_power_w: u32,
    pub sm_clock_base_mhz: u32,
    pub sm_clock_boost_mhz: u32,
    pub mem_clock_mhz: u32,
    pub temp_idle_c: u32,
    pub temp_max_c: u32,
}

pub fn lookup_gpu(name: &str) -> Option<GpuConfig> {
    let cfg = match name {
        "rtx6000" => GpuConfig {
            model: "NVIDIA RTX 6000 Ada Generation",
            memory_gb: 48,
            max_power_w: 300,
            idle_power_w: 30,
            sm_clock_base_mhz: 915,
            sm_clock_boost_mhz: 2505,
            mem_clock_mhz: 9001,
            temp_idle_c: 35,
            temp_max_c: 83,
        },
        "a100" => GpuConfig {
            model: "NVIDIA A100-SXM4-40GB",
            memory_gb: 40,
            max_power_w: 400,
            idle_power_w: 50,
            sm_clock_base_mhz: 765,
            sm_clock_boost_mhz: 1410,
            mem_clock_mhz: 1215,
            temp_idle_c: 40,
            temp_max_c: 85,
        },
        "h100" => GpuConfig {
            model: "NVIDIA H100 PCIe",
            memory_gb: 80,
            max_power_w: 350,
            idle_power_w: 40,
            sm_clock_base_mhz: 1095,
            sm_clock_boost_mhz: 1830,
            mem_clock_mhz: 1593,
            temp_idle_c: 35,
            temp_max_c: 82,
        },
        "h100-sxm" => GpuConfig {
            model: "NVIDIA H100 SXM5",
            memory_gb: 80,
            max_power_w: 700,
            idle_power_w: 70,
            sm_clock_base_mhz: 1350,
            sm_clock_boost_mhz: 1980,
            mem_clock_mhz: 1593,
            temp_idle_c: 40,
            temp_max_c: 88,
        },
        "h200" => GpuConfig {
            model: "NVIDIA H200",
            memory_gb: 141,
            max_power_w: 700,
            idle_power_w: 70,
            sm_clock_base_mhz: 1350,
            sm_clock_boost_mhz: 1980,
            mem_clock_mhz: 1593,
            temp_idle_c: 40,
            temp_max_c: 88,
        },
        "b200" => GpuConfig {
            model: "NVIDIA B200",
            memory_gb: 192,
            max_power_w: 1200,
            idle_power_w: 100,
            sm_clock_base_mhz: 1500,
            sm_clock_boost_mhz: 2200,
            mem_clock_mhz: 2000,
            temp_idle_c: 40,
            temp_max_c: 90,
        },
        "gb200" => GpuConfig {
            model: "NVIDIA GB200",
            memory_gb: 192,
            max_power_w: 1200,
            idle_power_w: 100,
            sm_clock_base_mhz: 1500,
            sm_clock_boost_mhz: 2200,
            mem_clock_mhz: 2000,
            temp_idle_c: 40,
            temp_max_c: 90,
        },
        _ => return None,
    };
    Some(cfg)
}

#[derive(Debug)]
struct FakeGpuState {
    idx: u32,
    cfg: GpuConfig,
    rng: RandomGenerator,
    load_offset: f64,
    uuid: String,
    mem_total: f64,
    power_limit: f64,
    pci_bus_id: String,
    device: String,

    util: f64,
    power: f64,
    temp: f64,
    mem_temp: f64,
    mem_used: f64,
    mem_free: f64,
    sm_clk: f64,
    mem_clk: f64,
    mem_copy: f64,
    enc_util: f64,
    dec_util: f64,
    sm_active: f64,
    energy: f64,
    power_viol: f64,
    thermal_viol: f64,
    xid: u64,
}

impl FakeGpuState {
    fn new(idx: u32, cfg: GpuConfig, mut rng: RandomGenerator, load_offset: f64) -> Self {
        let mut draw = |lo: u64, hi: u64| {
            rng.randrange_u64(lo, hi)
                .expect("dcgm uuid ranges are non-empty")
        };
        let uuid = format!(
            "GPU-{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
            draw(10u64.pow(8), 10u64.pow(9)),
            draw(0, 0x1_0000),
            draw(0, 0x1_0000),
            draw(0, 0x1_0000),
            draw(0, 10u64.pow(12)),
        );
        let mem_total = (cfg.memory_gb as f64) * 1024.0;
        let power_limit = cfg.max_power_w as f64;
        let pci_bus_id = format!("00000000:{:02x}:00.0", idx + 2);
        let device = format!("nvidia{idx}");
        FakeGpuState {
            idx,
            cfg,
            rng,
            load_offset,
            uuid,
            mem_total,
            power_limit,
            pci_bus_id,
            device,
            util: 0.0,
            power: 0.0,
            temp: 0.0,
            mem_temp: 0.0,
            mem_used: 0.0,
            mem_free: 0.0,
            sm_clk: 0.0,
            mem_clk: 0.0,
            mem_copy: 0.0,
            enc_util: 0.0,
            dec_util: 0.0,
            sm_active: 0.0,
            energy: 0.0,
            power_viol: 0.0,
            thermal_viol: 0.0,
            xid: 0,
        }
    }

    fn noise(&mut self, val: f64, variance: f64, max_val: f64) -> f64 {
        let factor: f64 = self.rng.uniform(1.0 - variance, 1.0 + variance);
        let noisy = val * factor;
        noisy.max(0.0).min(max_val)
    }

    fn update(&mut self, base_load: f64) {
        let load = (base_load + self.load_offset).clamp(0.0, 1.0);
        // Snapshot the config fields we need so we can hold a mutable borrow of rng/self.
        let idle_power_w = self.cfg.idle_power_w as f64;
        let max_power_w = self.cfg.max_power_w as f64;
        let temp_idle_c = self.cfg.temp_idle_c as f64;
        let temp_max_c = self.cfg.temp_max_c as f64;
        let sm_base = self.cfg.sm_clock_base_mhz as f64;
        let sm_boost = self.cfg.sm_clock_boost_mhz as f64;
        let mem_clock = self.cfg.mem_clock_mhz as f64;
        let memory_gb = self.cfg.memory_gb as f64;

        self.util = self.noise(5.0 + load * 95.0, 0.03, 100.0);
        self.power = self.noise(
            idle_power_w + load * (max_power_w - idle_power_w),
            0.02,
            max_power_w,
        );
        self.temp = self.noise(
            temp_idle_c + load * (temp_max_c - temp_idle_c),
            0.01,
            temp_max_c,
        );
        let mem_temp_off: f64 = self.rng.uniform(3.0, 8.0);
        self.mem_temp = (self.temp + mem_temp_off).min(temp_max_c + 10.0);
        self.sm_clk = self.noise(sm_base + load * (sm_boost - sm_base), 0.01, sm_boost);
        self.mem_clk = self.noise(mem_clock, 0.005, mem_clock);
        self.mem_used = self.noise(
            memory_gb * 1024.0 * (0.10 + load * 0.75),
            0.02,
            self.mem_total,
        );
        self.mem_free = self.mem_total - self.mem_used;
        self.mem_copy = self.noise(load * 50.0, 0.05, 100.0);
        // Video encode/decode engines: modest load-driven utilization (percent).
        self.enc_util = self.noise(load * 40.0, 0.05, 100.0);
        self.dec_util = self.noise(load * 30.0, 0.05, 100.0);
        // SM activity is a DCGM profiling ratio in [0, 1]; the runner scales it x100.
        self.sm_active = self.noise(0.05 + load * 0.90, 0.02, 1.0);

        self.energy += self.power * 1000.0; // 1 tick = 1 s
        if self.rng.random() < 0.0001 {
            self.xid += 1;
        }

        if self.power > max_power_w * 0.95 {
            let factor: f64 = self.rng.uniform(500.0, 2000.0);
            self.power_viol += (self.power - max_power_w * 0.95) / (max_power_w * 0.95) * factor;
        }
        if self.temp > temp_max_c - 5.0 {
            let factor: f64 = self.rng.uniform(100.0, 500.0);
            self.thermal_viol += (self.temp - (temp_max_c - 5.0)) * factor;
        }
    }
}

struct Inner {
    cfg: GpuConfig,
    hostname: String,
    load: f64,
    gpus: Vec<FakeGpuState>,
}

pub struct DcgmFaker {
    inner: Mutex<Inner>,
}

impl DcgmFaker {
    pub fn new(
        gpu_name: &str,
        num_gpus: u32,
        seed: Option<u64>,
        hostname: &str,
    ) -> Result<Self, String> {
        let cfg = lookup_gpu(gpu_name).ok_or_else(|| format!("Invalid GPU name: {gpu_name}"))?;
        Self::with_initial_load(cfg, num_gpus, seed, hostname, 0.0)
    }

    fn with_initial_load(
        cfg: GpuConfig,
        num_gpus: u32,
        seed: Option<u64>,
        hostname: &str,
        initial_load: f64,
    ) -> Result<Self, String> {
        let mut rng = RandomGenerator::from_seed(seed);
        let mut gpus = Vec::with_capacity(num_gpus as usize);
        for i in 0..num_gpus {
            let load_offset: f64 = rng.uniform(-0.05, 0.05);
            let gpu_seed = rng.random_u64();
            let gpu_rng = RandomGenerator::from_seed(Some(gpu_seed));
            gpus.push(FakeGpuState::new(i, cfg.clone(), gpu_rng, load_offset));
        }
        Ok(DcgmFaker {
            inner: Mutex::new(Inner {
                cfg,
                hostname: hostname.to_string(),
                load: initial_load.clamp(0.0, 1.0),
                gpus,
            }),
        })
    }

    pub fn set_load(&self, load: f64) {
        let mut g = self.inner.lock();
        g.load = load.clamp(0.0, 1.0);
    }

    pub fn generate(&self) -> String {
        let mut g = self.inner.lock();
        let load = g.load;
        for gpu in g.gpus.iter_mut() {
            gpu.update(load);
        }

        let mut out = String::new();
        let mappings: [(&str, &str, &str); 18] = [
            ("DCGM_FI_DEV_GPU_UTIL", "GPU utilization (in %).", "util"),
            ("DCGM_FI_DEV_POWER_USAGE", "Power draw (in W).", "power"),
            (
                "DCGM_FI_DEV_POWER_MGMT_LIMIT",
                "Power management limit (in W).",
                "power_limit",
            ),
            (
                "DCGM_FI_DEV_FB_USED",
                "Framebuffer memory used (in MiB).",
                "mem_used",
            ),
            (
                "DCGM_FI_DEV_FB_TOTAL",
                "Framebuffer memory total (in MiB).",
                "mem_total",
            ),
            (
                "DCGM_FI_DEV_FB_FREE",
                "Framebuffer memory free (in MiB).",
                "mem_free",
            ),
            ("DCGM_FI_DEV_GPU_TEMP", "GPU temperature (in C).", "temp"),
            (
                "DCGM_FI_DEV_MEMORY_TEMP",
                "Memory temperature (in C).",
                "mem_temp",
            ),
            (
                "DCGM_FI_DEV_SM_CLOCK",
                "SM clock frequency (in MHz).",
                "sm_clk",
            ),
            (
                "DCGM_FI_DEV_MEM_CLOCK",
                "Memory clock frequency (in MHz).",
                "mem_clk",
            ),
            (
                "DCGM_FI_DEV_MEM_COPY_UTIL",
                "Memory copy utilization (in %).",
                "mem_copy",
            ),
            (
                "DCGM_FI_DEV_ENC_UTIL",
                "Encoder utilization (in %).",
                "enc_util",
            ),
            (
                "DCGM_FI_DEV_DEC_UTIL",
                "Decoder utilization (in %).",
                "dec_util",
            ),
            (
                "DCGM_FI_PROF_SM_ACTIVE",
                "Ratio of cycles an SM has at least one warp assigned (0..1).",
                "sm_active",
            ),
            (
                "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
                "Total energy consumption since boot (in mJ).",
                "energy",
            ),
            ("DCGM_FI_DEV_XID_ERRORS", "XID error count.", "xid"),
            (
                "DCGM_FI_DEV_POWER_VIOLATION",
                "Power violation duration (in us).",
                "power_viol",
            ),
            (
                "DCGM_FI_DEV_THERMAL_VIOLATION",
                "Thermal violation duration (in us).",
                "thermal_viol",
            ),
        ];

        for (i, (name, help, attr)) in mappings.iter().enumerate() {
            writeln!(out, "# HELP {name} {help}").unwrap();
            writeln!(out, "# TYPE {name} gauge").unwrap();
            for gpu in g.gpus.iter() {
                let val: f64 = match *attr {
                    "util" => gpu.util,
                    "power" => gpu.power,
                    "power_limit" => gpu.power_limit,
                    "mem_used" => gpu.mem_used,
                    "mem_total" => gpu.mem_total,
                    "mem_free" => gpu.mem_free,
                    "temp" => gpu.temp,
                    "mem_temp" => gpu.mem_temp,
                    "sm_clk" => gpu.sm_clk,
                    "mem_clk" => gpu.mem_clk,
                    "mem_copy" => gpu.mem_copy,
                    "enc_util" => gpu.enc_util,
                    "dec_util" => gpu.dec_util,
                    "sm_active" => gpu.sm_active,
                    "energy" => gpu.energy,
                    "xid" => gpu.xid as f64,
                    "power_viol" => gpu.power_viol,
                    "thermal_viol" => gpu.thermal_viol,
                    _ => 0.0,
                };
                writeln!(
                    out,
                    "{name}{{gpu=\"{idx}\",UUID=\"{uuid}\",pci_bus_id=\"{pci}\",device=\"{dev}\",modelName=\"{model}\",Hostname=\"{host}\"}} {val:.2}",
                    idx = gpu.idx,
                    uuid = gpu.uuid,
                    pci = gpu.pci_bus_id,
                    dev = gpu.device,
                    model = g.cfg.model,
                    host = g.hostname,
                    val = val,
                )
                .unwrap();
            }
            // No blank line between metric blocks - matches Python's "\n".join
            let _ = i;
        }
        out
    }
}

/// A pool of DCGM fakers (one per instance). Index 0 = /dcgm1/metrics, etc.
pub struct DcgmPool {
    fakers: Vec<DcgmFaker>,
    request_counter: AtomicU32,
}

impl DcgmPool {
    pub fn new(fakers: Vec<DcgmFaker>) -> Self {
        Self {
            fakers,
            request_counter: AtomicU32::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.fakers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fakers.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&DcgmFaker> {
        self.fakers.get(index)
    }

    pub fn set_load(&self, load: f64) {
        for f in &self.fakers {
            f.set_load(load);
        }
    }

    pub fn inc_requests(&self) {
        self.request_counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn request_count(&self) -> u32 {
        self.request_counter.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_all_metric_families() {
        let faker = DcgmFaker::new("h200", 2, Some(42), "localhost").unwrap();
        let out = faker.generate();
        assert!(out.contains("DCGM_FI_DEV_GPU_UTIL"));
        assert!(out.contains("DCGM_FI_DEV_POWER_USAGE"));
        assert!(out.contains("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"));
        assert!(out.contains("DCGM_FI_DEV_XID_ERRORS"));
        // Encode/decode/SM-activity fills consumed by the runner GPU telemetry decoder.
        assert!(out.contains("DCGM_FI_DEV_ENC_UTIL"));
        assert!(out.contains("DCGM_FI_DEV_DEC_UTIL"));
        assert!(out.contains("DCGM_FI_PROF_SM_ACTIVE"));
        assert!(out.contains("NVIDIA H200"));
        assert!(out.contains("modelName=\"NVIDIA H200\""));
        assert!(out.contains("gpu=\"0\""));
        assert!(out.contains("gpu=\"1\""));
    }

    #[test]
    fn invalid_gpu_name_fails() {
        assert!(DcgmFaker::new("unknown", 1, Some(1), "localhost").is_err());
    }

    #[test]
    fn seeded_output_is_deterministic() {
        let a = DcgmFaker::new("a100", 1, Some(7), "h1").unwrap();
        a.set_load(0.5);
        let a_out = a.generate();
        let b = DcgmFaker::new("a100", 1, Some(7), "h1").unwrap();
        b.set_load(0.5);
        let b_out = b.generate();
        assert_eq!(a_out, b_out);
    }

    #[test]
    fn load_affects_utilization() {
        let faker = DcgmFaker::new("h100", 1, Some(99), "h").unwrap();
        faker.set_load(0.0);
        let idle = faker.generate();
        faker.set_load(1.0);
        let busy = faker.generate();
        // Very loose heuristic: idle util should be smaller than busy util
        let util_idle: f64 = extract_metric(&idle, "DCGM_FI_DEV_GPU_UTIL").unwrap();
        let util_busy: f64 = extract_metric(&busy, "DCGM_FI_DEV_GPU_UTIL").unwrap();
        assert!(util_idle < util_busy, "idle={util_idle} busy={util_busy}");
        // SM_ACTIVE is a 0..1 ratio the runner scales x100.
        let sm_busy: f64 = extract_metric(&busy, "DCGM_FI_PROF_SM_ACTIVE").unwrap();
        assert!(
            (0.0..=1.0).contains(&sm_busy),
            "sm_active out of [0,1]: {sm_busy}"
        );
        let sm_idle: f64 = extract_metric(&idle, "DCGM_FI_PROF_SM_ACTIVE").unwrap();
        assert!(sm_idle < sm_busy, "idle={sm_idle} busy={sm_busy}");
    }

    fn extract_metric(text: &str, name: &str) -> Option<f64> {
        for line in text.lines() {
            if line.starts_with(name)
                && line.contains('{')
                && let Some(value) = line.split_whitespace().last()
                && let Ok(v) = value.parse::<f64>()
            {
                return Some(v);
            }
        }
        None
    }
}
