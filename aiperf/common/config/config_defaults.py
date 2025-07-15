# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.enums import (
    AudioFormat,
    CommunicationBackend,
    CustomDatasetType,
    EndpointType,
    ImageFormat,
    ModelSelectionStrategy,
    RequestRateMode,
    ServiceRunType,
    TimingMode,
)
from aiperf.progress.progress_models import (
    SweepCompletionTrigger,
    SweepMultiParamOrder,
    SweepParamOrder,
)


#
# Config Defaults
@dataclass(frozen=True)
class UserDefaults:
    MODEL_NAMES = None  # This should be set by the user
    VERBOSE = False
    TEMPLATE_FILENAME = "aiperf_config.yaml"


@dataclass(frozen=True)
class EndPointDefaults:
    MODEL_SELECTION_STRATEGY = ModelSelectionStrategy.ROUND_ROBIN
    CUSTOM = None
    TYPE = EndpointType.OPENAI_CHAT_COMPLETIONS
    STREAMING = True
    SERVER_METRICS_URLS = ["http://localhost:8002/metrics"]
    URL = "localhost:8080"
    GRPC_METHOD = ""
    TIMEOUT = 30.0
    API_KEY = None


@dataclass(frozen=True)
class InputDefaults:
    BATCH_SIZE = 1
    EXTRA = {}
    GOODPUT = {}
    HEADERS = {}
    FILE = None
    CUSTOM_DATASET_TYPE = CustomDatasetType.TRACE
    RANDOM_SEED = None
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


@dataclass(frozen=True)
class ServiceDefaults:
    SERVICE_RUN_TYPE = ServiceRunType.MULTIPROCESSING
    COMM_BACKEND = CommunicationBackend.ZMQ_IPC
    COMM_CONFIG = None
    HEARTBEAT_TIMEOUT = 60.0
    REGISTRATION_TIMEOUT = 60.0
    COMMAND_TIMEOUT = 10.0
    HEARTBEAT_INTERVAL = 1.0
    MIN_WORKERS = None
    MAX_WORKERS = None
    LOG_LEVEL = "INFO"
    DISABLE_UI = False
    ENABLE_UVLOOP = True
    RESULT_PARSER_SERVICE_COUNT = 2
    DEBUG_SERVICES = None


@dataclass(frozen=True)
class LoadGeneratorDefaults:
    CONCURRENCY = 1
    REQUEST_RATE = None
    REQUEST_COUNT = 10
    WARMUP_REQUEST_COUNT = 0
    CONCURRENCY_RAMP_UP_TIME = None
    REQUEST_RATE_MODE = RequestRateMode.POISSON
    TIMING_MODE = TimingMode.CONCURRENCY


@dataclass(frozen=True)
class MeasurementDefaults:
    MEASUREMENT_INTERVAL = 10_000
    STABILITY_PERCENTAGE = 0.95


@dataclass(frozen=True)
class SweepParamDefaults:
    VALUES = None
    ORDER = SweepParamOrder.ASCENDING
    COMPLETION_TRIGGER = SweepCompletionTrigger.COMPLETED_PROFILES
    START = None
    STEP = None
    END = None
    MAX_PROFILES = None


@dataclass(frozen=True)
class SweepDefaults:
    PARAMS = None
    ORDER = SweepMultiParamOrder.DEPTH_FIRST
