# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.enums import (
    AudioFormat,
    CustomDatasetType,
    ImageFormat,
    InferenceClientType,
    ModelSelectionStrategy,
    RequestPayloadType,
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
    REQUEST_PAYLOAD_TYPE = RequestPayloadType.OPENAI_CHAT_COMPLETIONS
    CUSTOM = ""
    TYPE = InferenceClientType.OPENAI
    STREAMING = True
    SERVER_METRICS_URLS = ["http://localhost:8002/metrics"]
    URL = "localhost:8080"
    GRPC_METHOD = ""


@dataclass(frozen=True)
class InputDefaults:
    EXTRA = ""
    GOODPUT = {}
    HEADER = ""
    FILE = None
    CUSTOM_DATASET_TYPE = CustomDatasetType.TRACE
    NUM_DATASET_ENTRIES = 100
    RANDOM_SEED = None


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
class PromptDefaults:
    BATCH_SIZE = 1


@dataclass(frozen=True)
class InputTokensDefaults:
    MEAN = 550
    STDDEV = 0.0
    BLOCK_SIZE = 512


@dataclass(frozen=True)
class OutputTokensDefaults:
    MEAN = 0
    DETERMINISTIC = False
    STDDEV = 0


@dataclass(frozen=True)
class PrefixPromptDefaults:
    POOL_SIZE = 0
    LENGTH = 0


@dataclass(frozen=True)
class ConversationDefaults:
    NUM = 100


@dataclass(frozen=True)
class TurnDefaults:
    MEAN = 1
    STDDEV = 0


@dataclass(frozen=True)
class TurnDelayDefaults:
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


@dataclass(frozen=True)
class OutputTokenDefaults:
    MEAN = None
    DETERMINISTIC = False
    STDDEV = 0
