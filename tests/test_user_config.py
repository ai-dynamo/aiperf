#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import ruamel.yaml
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str

import sys

import unittest

from unittest.mock import patch

# # Skip type checking to avoid mypy error
# # Issue: https://github.com/python/mypy/issues/10632
# import yaml  # type: ignore

from aiperf.common.models.config.user_config import UserConfig
import pytest
from unittest.mock import patch


class TestUserConfig:
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        yield
        patch.stopall()

    ##########################################################################
    # Test Model Name
    ##########################################################################
    def test_yaml_writer(self, tmp_path):
        """
        Test that a configuration with a model name is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            # Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.
            model_names: ["model1", "model2"]
            
            endpoint:
                model_selection_strategy: random
                backend: VLLM
                custom: custom_endpoint
                type: kserve
                streaming: True
                server_metrics_urls: "http://test_server_metrics_url:8002/metrics"
                url: "test_url"
                grpc_method: "test_grpc_method"
            """)
        # yapf: enable

        config = parse_yaml_raw_as(UserConfig, yaml_str)

        yaml_str = config.serialize_to_yaml()
        test_file = tmp_path / "test_no_verbose.yaml"
        with open(test_file, "w") as f:
            f.write(yaml_str)

        assert config.model_names == "model1, model2"

    def test_yaml_writer_verbose(self, tmp_path):
        """
        Test that a configuration with a model name is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            # Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.
            model_names: model1, model2
            verbose: true
            """)
        # yapf: enable

        config = parse_yaml_raw_as(UserConfig, yaml_str)

        yaml_str = config.serialize_to_yaml()
        test_file = tmp_path / "test_verbose.yaml"
        with open(test_file, "w") as f:
            f.write(yaml_str)

        assert config.model_names == "model1, model2"
