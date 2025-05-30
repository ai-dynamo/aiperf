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

from enum import Enum

from pydantic.dataclasses import dataclass


class CaseInsensitiveEnum(str, Enum):
    """
    A custom enumeration class that extends `str` and `Enum` to provide case-insensitive
    lookup for its members. This allows string values to match enumeration members
    regardless of their case.
    Methods:
        _missing_(cls, value):
            A class method that is called when a value is not found in the enumeration.
            If the value is a string, it performs a case-insensitive comparison with
            the enumeration members and returns the matching member if found. If no
            match is found, it returns `None`.
    """

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


#
# Enums
class OutputFormat(CaseInsensitiveEnum):
    TENSORRTLLM = "TENSORRTLLM"
    VLLM = "VLLM"


class ModelSelectionStrategy(CaseInsensitiveEnum):
    ROUND_ROBIN = "ROUND_ROBIN"
    RANDOM = "RANDOM"


class AudioFormat(CaseInsensitiveEnum):
    WAV = "WAV"
    MP3 = "MP3"


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
