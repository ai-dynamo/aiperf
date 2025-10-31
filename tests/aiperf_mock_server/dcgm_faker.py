# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DCGM faker that generates realistic load-driven GPU metrics."""

import random
from dataclasses import dataclass


@dataclass
class GPUConfig:
    model: str
    memory_gb: int
    max_power_w: int
    idle_power_w: int
    sm_clock_base_mhz: int
    sm_clock_boost_mhz: int
    mem_clock_mhz: int
    temp_idle_c: int
    temp_max_c: int


GPU_CONFIGS = {
    "rtx6000": GPUConfig("NVIDIA RTX 6000 Ada Generation", 48, 300, 30, 915, 2505, 9001, 35, 83),
    "a100": GPUConfig("NVIDIA A100-SXM4-40GB", 40, 400, 50, 765, 1410, 1215, 40, 85),
    "h100": GPUConfig("NVIDIA H100 PCIe", 80, 350, 40, 1095, 1830, 1593, 35, 82),
    "h100-sxm": GPUConfig("NVIDIA H100 SXM5", 80, 700, 70, 1350, 1980, 1593, 40, 88),
    "h200": GPUConfig("NVIDIA H200", 141, 700, 70, 1350, 1980, 1593, 40, 88),
    "b200": GPUConfig("NVIDIA B200", 192, 1200, 100, 1500, 2200, 2000, 40, 90),
    "gb200": GPUConfig("NVIDIA GB200", 192, 1200, 100, 1500, 2200, 2000, 40, 90),
}  # fmt: skip
"""GPU configuration for realistic metrics generation."""

METRIC_MAPPINGS = [
    ("DCGM_FI_DEV_GPU_UTIL", "GPU utilization (in %).", "util"),
    ("DCGM_FI_DEV_POWER_USAGE", "Power draw (in W).", "power"),
    ("DCGM_FI_DEV_POWER_MGMT_LIMIT", "Power management limit (in W).", "power_limit"),
    ("DCGM_FI_DEV_FB_USED", "Framebuffer memory used (in MiB).", "mem_used"),
    ("DCGM_FI_DEV_FB_TOTAL", "Framebuffer memory total (in MiB).", "mem_total"),
    ("DCGM_FI_DEV_FB_FREE", "Framebuffer memory free (in MiB).", "mem_free"),
    ("DCGM_FI_DEV_GPU_TEMP", "GPU temperature (in C).", "temp"),
    ("DCGM_FI_DEV_MEMORY_TEMP", "Memory temperature (in C).", "mem_temp"),
    ("DCGM_FI_DEV_SM_CLOCK", "SM clock frequency (in MHz).", "sm_clk"),
    ("DCGM_FI_DEV_MEM_CLOCK", "Memory clock frequency (in MHz).", "mem_clk"),
    ("DCGM_FI_DEV_MEM_COPY_UTIL", "Memory copy utilization (in %).", "mem_copy"),
    ("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", "Total energy consumption since boot (in mJ).", "energy"),
    ("DCGM_FI_DEV_XID_ERRORS", "XID error count.", "xid"),
    ("DCGM_FI_DEV_POWER_VIOLATION", "Power violation duration (in us).", "power_viol"),
    ("DCGM_FI_DEV_THERMAL_VIOLATION", "Thermal violation duration (in us).", "thermal_viol"),
]  # fmt: skip
"""Metric mappings for DCGM metrics to internal FakeGPU attributes."""


@dataclass
class FakeGPU:
    """Single GPU state and metrics."""

    idx: int
    cfg: GPUConfig
    rng: random.Random
    load_offset: float

    # Default values for metrics
    energy: float = 0.0
    power_viol: float = 0.0
    thermal_viol: float = 0.0
    xid: int = 0
    util: float = 0.0
    power: float = 0.0
    temp: float = 0.0
    mem_temp: float = 0.0
    mem_used: float = 0.0
    mem_free: float = 0.0
    sm_clk: float = 0.0
    mem_clk: float = 0.0
    mem_copy: float = 0.0

    def __post_init__(self):
        self.uuid = f"GPU-{self.rng.randint(10**8, 10**9):08x}-{self.rng.randint(0, 0xFFFF):04x}-{self.rng.randint(0, 0xFFFF):04x}-{self.rng.randint(0, 0xFFFF):04x}-{self.rng.randint(0, 10**12):012x}"
        self.mem_total = float(self.cfg.memory_gb * 1024)
        self.power_limit = float(self.cfg.max_power_w)
        self.pci_bus_id = f"00000000:{self.idx + 2:02x}:00.0"
        self.device = f"nvidia{self.idx}"

    def _noise(self, val: float, variance: float, max_val: float) -> float:
        """Add noise to a value and clamp to [0, max]."""
        noisy = val * self.rng.uniform(1 - variance, 1 + variance)
        return max(0.0, min(noisy, max_val))

    def update(self, base_load: float) -> None:
        """Update all metrics based on current load (computed once)."""
        load = max(0.0, min(1.0, base_load + self.load_offset))
        c = self.cfg

        # Update all current metrics
        self.util = self._noise(5 + load * 95, 0.03, 100.0)
        self.power = self._noise(c.idle_power_w + load * (c.max_power_w - c.idle_power_w), 0.02, c.max_power_w)  # fmt: skip
        self.temp = self._noise(c.temp_idle_c + load * (c.temp_max_c - c.temp_idle_c), 0.01, c.temp_max_c)  # fmt: skip
        self.mem_temp = min(self.temp + self.rng.uniform(3, 8), c.temp_max_c + 10)
        self.sm_clk = self._noise(c.sm_clock_base_mhz + load * (c.sm_clock_boost_mhz - c.sm_clock_base_mhz), 0.01, c.sm_clock_boost_mhz)  # fmt: skip
        self.mem_clk = self._noise(c.mem_clock_mhz, 0.005, c.mem_clock_mhz)
        self.mem_used = self._noise(c.memory_gb * 1024 * (0.10 + load * 0.75), 0.02, self.mem_total)  # fmt: skip
        self.mem_free = self.mem_total - self.mem_used
        self.mem_copy = self._noise(load * 50, 0.05, 100.0)

        # Update cumulative metrics
        self.energy += self.power * 1000  # 1 tick = 1s
        if self.rng.random() < 0.0001:  # 1 in 10,000 chance of XID error
            self.xid += 1

        # Update violations
        if self.power > c.max_power_w * 0.95:
            self.power_viol += (self.power - c.max_power_w * 0.95) / (c.max_power_w * 0.95) * self.rng.uniform(500, 2000)  # fmt: skip
        if self.temp > c.temp_max_c - 5:
            self.thermal_viol += (self.temp - (c.temp_max_c - 5)) * self.rng.uniform(100, 500)  # fmt: skip


class DCGMFaker:
    """Simulated DCGM Prometheus metrics generator."""

    def __init__(self, gpu_name: str = "rtx6000", num_gpus: int = 2, seed: int | None = None, hostname: str = "localhost", initial_load: float = 0.7):  # fmt: skip
        """Initialize faker with load level (0.0=idle, 1.0=max)."""
        if gpu_name not in GPU_CONFIGS:
            raise ValueError(f"Invalid GPU name: {gpu_name}")
        self.cfg = GPU_CONFIGS[gpu_name]
        self.hostname = hostname
        self.load = max(0.0, min(1.0, initial_load))
        self.rng = random.Random(seed)
        self.gpus = [FakeGPU(i, self.cfg, self.rng, self.rng.uniform(-0.05, 0.05)) for i in range(num_gpus)]  # fmt: skip

    def set_load(self, load: float) -> None:
        """Set load level (0.0=idle, 1.0=max). Affects all metrics."""
        self.load = max(0.0, min(1.0, load))

    def _format_metric(self, name: str, help_text: str, attr: str) -> str:
        """Format Prometheus metric block."""
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} gauge"]
        for gpu in self.gpus:
            lines.append(
                f'{name}{{gpu="{gpu.idx}",UUID="{gpu.uuid}",pci_bus_id="{gpu.pci_bus_id}",device="{gpu.device}",modelName="{self.cfg.model}",Hostname="{self.hostname}"}} {float(getattr(gpu, attr)):.2f}'
            )
        return "\n".join(lines)

    def generate(self) -> str:
        """Generate complete DCGM metrics snapshot based on current load."""
        for gpu in self.gpus:
            gpu.update(self.load)
        metrics = [self._format_metric(*metric) for metric in METRIC_MAPPINGS]
        return "\n".join(metrics) + "\n"
