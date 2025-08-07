# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
import time
import subprocess


class TestAIPerfMockServerIntegration:
    @pytest.fixture(scope="class")
    def server_fixture(self):
        """Start a test LLM inference server for integration testing"""
        # Start test server (e.g., vLLM, Triton, etc.)
        server_process = subprocess.Popen(
            [
                "aiperf-mock-server",
                "-m",
                "Qwen/Qwen3-0.6B",
                "-p",
                "8080",
            ]
        )

        # Wait for server to be ready
        self._wait_for_server_ready("http://localhost:8080")

        yield "http://localhost:8080"

        # Cleanup
        server_process.terminate()
        server_process.wait()

    def _wait_for_server_ready(self, url, timeout=60):
        """Wait for server to become ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        raise RuntimeError("Server failed to start within timeout")

    def test_basic_performance_analysis(self, test_server):
        """Test basic throughput and latency measurement"""
        # Run your CLI tool against the test server
        result = subprocess.run(
            [
                "aiperf",
                "profile",
                "--model-names",
                "Qwen/Qwen3-0.6B",
                "--concurrency",
                "1",
                "--request-count",
                "1",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"aiperf failed: {result.stderr}"
