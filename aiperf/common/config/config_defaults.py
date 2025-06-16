#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.enums import (
    AudioFormat,
    ImageFormat,
    ModelSelectionStrategy,
    OutputFormat,
)


#
# Config Defaults
@dataclass(frozen=True)
class UserDefaults:
    MODEL_NAMES = []
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
    LENGTH_MEAN = 0.0
    LENGTH_STDDEV = 0.0
    FORMAT = AudioFormat.WAV
    DEPTHS = [16]
    SAMPLE_RATES = [16.0]
    NUM_CHANNELS = 1


@dataclass(frozen=True)
class ImageDefaults:
    BATCH_SIZE = 1
    WIDTH_MEAN = 0.0
    WIDTH_STDDEV = 0.0
    HEIGHT_MEAN = 0.0
    HEIGHT_STDDEV = 0.0
    FORMAT = ImageFormat.PNG


@dataclass(frozen=True)
class InputTokensDefaults:
    MEAN = 550
    STDDEV = 0.0


@dataclass(frozen=True)
class OutputTokensDefaults:
    MEAN = 0
    DETERMINISTIC = False
    STDDEV = 0


@dataclass(frozen=True)
class PrefixPromptDefaults:
    POOL_SIZE = 0
    LENGTH = 100


@dataclass(frozen=True)
class SessionsDefaults:
    NUM = 0


@dataclass(frozen=True)
class SessionTurnsDefaults:
    MEAN = 1.0
    STDDEV = 0.0


@dataclass(frozen=True)
class SessionTurnDelayDefaults:
    MEAN = 0.0
    STDDEV = 0.0
    RATIO = 1.0


@dataclass(frozen=True)
class OutputDefaults:
    ARTIFACT_DIRECTORY = Path("./artifacts")


@dataclass(frozen=True)
class TokenizerDefaults:
    NAME = ""
    REVISION = "main"
    TRUST_REMOTE_CODE = False
