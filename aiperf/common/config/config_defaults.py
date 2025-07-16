# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.enums import (
    AIPerfLogLevel,
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


#
# Config Defaults
@dataclass(frozen=True)
class UserDefaults:
    MODEL_NAMES = None  # This must be set by the user


@dataclass(frozen=True)
class CLIDefaults:
    TEMPLATE_FILENAME = "aiperf_config.yaml"


@dataclass(frozen=True)
class EndPointDefaults:
    MODEL_SELECTION_STRATEGY = ModelSelectionStrategy.ROUND_ROBIN
    CUSTOM_ENDPOINT = None
    TYPE = EndpointType.OPENAI_CHAT_COMPLETIONS
    STREAMING = True
    SERVER_METRICS_URLS = ["http://localhost:8002/metrics"]
    URL = "localhost:8080"
    GRPC_METHOD = ""
    TIMEOUT = 600.0
    API_KEY = None


@dataclass(frozen=True)
class InputDefaults:
    BATCH_SIZE = 1
    EXTRA = {}
    GOODPUT = {}
    HEADERS = {}
    FILE = None
    FIXED_SCHEDULE = False
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
    FORMAT = ImageFormat.PNG  # TODO: GenAI-Perf was RANDOM default?


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
    PROFILE_EXPORT_FILE = Path("profile_export.json")


@dataclass(frozen=True)
class TokenizerDefaults:
    NAME = None
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
    HEARTBEAT_INTERVAL_SECONDS = 5.0
    LOG_LEVEL = AIPerfLogLevel.INFO
    VERBOSE = False
    EXTRA_VERBOSE = False
    LOG_PATH = None
    DISABLE_UI = False
    ENABLE_UVLOOP = True
    RESULT_PARSER_SERVICE_COUNT = 2
    ENABLE_YAPPI = False
    DEBUG_SERVICES = None
    TRACE_SERVICES = None
    PROGRESS_REPORT_INTERVAL_SECONDS = 1.0


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


# TODO: Enable this when we implement the sweep command
# @dataclass(frozen=True)
# class SweepParamDefaults:
#     VALUES = None
#     ORDER = SweepParamOrder.ASCENDING
#     COMPLETION_TRIGGER = SweepCompletionTrigger.COMPLETED_PROFILES
#     START = None
#     STEP = None
#     END = None
#     MAX_PROFILES = None


# TODO: Enable this when we implement the sweep command
# @dataclass(frozen=True)
# class SweepDefaults:
#     PARAMS = None
#     ORDER = SweepMultiParamOrder.DEPTH_FIRST
