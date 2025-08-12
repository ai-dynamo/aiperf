# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.enums import (
    AIPerfLogLevel,
    AIPerfUIType,
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
class CLIDefaults:
    TEMPLATE_FILENAME = "aiperf_config.yaml"


@dataclass(frozen=True)
class EndpointDefaults:
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
    HEADERS = {}
    FILE = None
    FIXED_SCHEDULE = False
    FIXED_SCHEDULE_AUTO_OFFSET = False
    FIXED_SCHEDULE_START_OFFSET = None
    FIXED_SCHEDULE_END_OFFSET = None
    CUSTOM_DATASET_TYPE = CustomDatasetType.MOONCAKE_TRACE
    RANDOM_SEED = None
    NUM_DATASET_ENTRIES = 100


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
    SHOW_INTERNAL_METRICS = False


@dataclass(frozen=True)
class TokenizerDefaults:
    NAME = None
    REVISION = "main"
    TRUST_REMOTE_CODE = False


@dataclass(frozen=True)
class OutputTokensDefaults:
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
    ENABLE_UVLOOP = True
    RECORD_PROCESSOR_SERVICE_COUNT = None
    ENABLE_YAPPI = False
    DEBUG_SERVICES = None
    TRACE_SERVICES = None
    PROGRESS_REPORT_INTERVAL = 1.0
    UI_TYPE = AIPerfUIType.DASHBOARD


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
class WorkersDefaults:
    MIN = None
    MAX = None
    HEALTH_CHECK_INTERVAL = 1.0
