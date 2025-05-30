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


from pydantic.dataclasses import dataclass

from aiperf.common.enums import AudioFormat, ModelSelectionStrategy, OutputFormat


#
# Config Defaults
@dataclass(frozen=True)
class UserDefaults:
    MODEL_NAMES = None
    VERBOSE = False
    TEMPLATE_FILENAME = "aiperf_config.yaml"


@dataclass(frozen=True)
class EndPointDefaults:
    MODEL_SELECTION_STRATEGY = ModelSelectionStrategy.ROUND_ROBIN
    BACKEND = OutputFormat.TENSORRTLLM
    CUSTOM = ""
    TYPE = "kserve"
    STREAMING = False
    SERVER_METRICS_URLS = ["http://localhost:8002/metrics"]
    URL = "localhost:8001"
    GRPC_METHOD = ""


@dataclass(frozen=True)
class InputDefaults:
    BATCH_SIZE = 1
    EXTRA = ""
    GOODPUT = {}
    HEADER = ""
    FILE = None
    NUM_DATASET_ENTRIES = 100
    RANDOM_SEED = 0


@dataclass(frozen=True)
class AudioDefaults:
    BATCH_SIZE = 1
    LENGTH_MEAN = 0
    LENGTH_STDDEV = 0
    FORMAT = AudioFormat.WAV
    DEPTHS = [16]
    SAMPLE_RATES = [16]
    NUM_CHANNELS = 1
