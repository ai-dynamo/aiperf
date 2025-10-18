# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dcgm_faker module."""

import pytest
from aiperf_mock_server.dcgm_faker import (
    GPU_CONFIGS,
    METRIC_MAPPINGS,
    DCGMFaker,
    FakeGPU,
    GPUConfig,
)


class TestGPUConfig:
    """Tests for GPUConfig dataclass."""

    @pytest.mark.parametrize(
        "gpu_name",
        ["rtx6000", "a100", "h100", "h100-sxm", "h200", "b200", "gb200"],
    )
    def test_all_gpu_configs_exist(self, gpu_name):
        assert gpu_name in GPU_CONFIGS
        config = GPU_CONFIGS[gpu_name]
        assert isinstance(config, GPUConfig)


class TestFakeGPU:
    """Tests for FakeGPU class."""

    @pytest.fixture
    def gpu_config(self):
        return GPU_CONFIGS["h200"]

    @pytest.fixture
    def fake_gpu(self, gpu_config):
        import random

        rng = random.Random(42)
        return FakeGPU(idx=0, cfg=gpu_config, rng=rng, load_offset=0.0)

    def test_fake_gpu_initialization(self, fake_gpu, gpu_config):
        assert fake_gpu.idx == 0
        assert fake_gpu.cfg == gpu_config
        assert fake_gpu.uuid.startswith("GPU-")
        assert fake_gpu.device == "nvidia0"
        assert fake_gpu.pci_bus_id == "00000000:02:00.0"
        assert fake_gpu.mem_total == gpu_config.memory_gb * 1024

    def test_fake_gpu_uuid_format(self, fake_gpu):
        uuid_parts = fake_gpu.uuid.split("-")
        assert len(uuid_parts) == 6
        assert uuid_parts[0] == "GPU"

    def test_update_idle_load(self, fake_gpu):
        fake_gpu.update(0.0)
        assert fake_gpu.util >= 0
        assert fake_gpu.util <= 100
        assert fake_gpu.power >= fake_gpu.cfg.idle_power_w * 0.9
        assert fake_gpu.temp >= fake_gpu.cfg.temp_idle_c * 0.9

    def test_update_high_load(self, fake_gpu):
        fake_gpu.update(1.0)
        assert fake_gpu.util > 50
        assert fake_gpu.power > fake_gpu.cfg.idle_power_w * 2
        assert fake_gpu.temp > fake_gpu.cfg.temp_idle_c + 10

    def test_update_metrics_in_range(self, fake_gpu):
        fake_gpu.update(0.7)
        assert 0 <= fake_gpu.util <= 100
        assert 0 <= fake_gpu.power <= fake_gpu.cfg.max_power_w * 1.1
        assert (
            fake_gpu.cfg.temp_idle_c <= fake_gpu.temp <= fake_gpu.cfg.temp_max_c * 1.1
        )
        assert 0 <= fake_gpu.mem_used <= fake_gpu.mem_total
        assert fake_gpu.mem_free >= 0

    def test_memory_consistency(self, fake_gpu):
        fake_gpu.update(0.5)
        assert abs((fake_gpu.mem_used + fake_gpu.mem_free) - fake_gpu.mem_total) < 0.01

    def test_cumulative_metrics_increase(self, fake_gpu):
        initial_energy = fake_gpu.energy
        fake_gpu.update(0.5)
        assert fake_gpu.energy > initial_energy

    def test_violations_at_high_load(self, fake_gpu):
        initial_power_viol = fake_gpu.power_viol
        initial_thermal_viol = fake_gpu.thermal_viol
        for _ in range(10):
            fake_gpu.update(1.0)
        assert fake_gpu.power_viol >= initial_power_viol
        assert fake_gpu.thermal_viol >= initial_thermal_viol


class TestDCGMFaker:
    """Tests for DCGMFaker class."""

    def test_initialization(self, dcgm_faker):
        assert dcgm_faker.cfg == GPU_CONFIGS["h200"]
        assert dcgm_faker.load == 0.7
        assert len(dcgm_faker.gpus) == 2

    def test_invalid_gpu_name(self):
        with pytest.raises(ValueError, match="Invalid GPU name"):
            DCGMFaker(gpu_name="invalid-gpu")

    def test_set_load(self, dcgm_faker):
        dcgm_faker.set_load(0.5)
        assert dcgm_faker.load == 0.5

    def test_set_load_clamps(self, dcgm_faker):
        dcgm_faker.set_load(1.5)
        assert dcgm_faker.load == 1.0
        dcgm_faker.set_load(-0.5)
        assert dcgm_faker.load == 0.0

    def test_generate_output_format(self, dcgm_faker):
        metrics = dcgm_faker.generate()
        assert isinstance(metrics, str)
        assert "# HELP" in metrics
        assert "# TYPE" in metrics
        assert "gauge" in metrics

    def test_generate_contains_all_metrics(self, dcgm_faker):
        metrics = dcgm_faker.generate()
        for metric_name, _, _ in METRIC_MAPPINGS:
            assert metric_name in metrics

    def test_generate_contains_gpu_labels(self, dcgm_faker):
        metrics = dcgm_faker.generate()
        assert 'gpu="0"' in metrics
        assert 'gpu="1"' in metrics
        assert f'modelName="{dcgm_faker.cfg.model}"' in metrics

    def test_generate_deterministic_with_seed(self):
        faker1 = DCGMFaker(gpu_name="h200", num_gpus=2, seed=123)
        faker2 = DCGMFaker(gpu_name="h200", num_gpus=2, seed=123)
        metrics1 = faker1.generate()
        metrics2 = faker2.generate()
        assert metrics1 == metrics2

    def test_generate_changes_with_load(self, dcgm_faker):
        dcgm_faker.set_load(0.2)
        metrics_low = dcgm_faker.generate()
        dcgm_faker.set_load(0.9)
        metrics_high = dcgm_faker.generate()
        assert metrics_low != metrics_high

    def test_multiple_gpus(self):
        faker = DCGMFaker(gpu_name="a100", num_gpus=4, seed=42)
        assert len(faker.gpus) == 4
        metrics = faker.generate()
        for i in range(4):
            assert f'gpu="{i}"' in metrics

    @pytest.mark.parametrize("num_gpus", [1, 2, 4, 8])
    def test_various_gpu_counts(self, num_gpus):
        faker = DCGMFaker(gpu_name="h100", num_gpus=num_gpus, seed=42)
        assert len(faker.gpus) == num_gpus

    def test_hostname_in_metrics(self):
        faker = DCGMFaker(gpu_name="h200", num_gpus=1, hostname="test-server")
        metrics = faker.generate()
        assert 'Hostname="test-server"' in metrics

    def test_initial_load_applied(self):
        faker_low = DCGMFaker(gpu_name="h200", num_gpus=1, seed=42, initial_load=0.1)
        faker_high = DCGMFaker(gpu_name="h200", num_gpus=1, seed=42, initial_load=0.9)

        metrics_low = faker_low.generate()
        metrics_high = faker_high.generate()

        assert metrics_low != metrics_high

    def test_gpu_load_offsets_create_variance(self):
        faker = DCGMFaker(gpu_name="h200", num_gpus=2, seed=42, initial_load=0.5)
        faker.generate()

        gpu0_util = faker.gpus[0].util
        gpu1_util = faker.gpus[1].util
        assert gpu0_util != gpu1_util
