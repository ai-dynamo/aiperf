# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.enums import (
    AIPerfLogLevel,
    AIPerfUIType,
    AudioFormat,
    CommunicationBackend,
    EndpointType,
    ExportLevel,
    ImageFormat,
    ModelSelectionStrategy,
    RequestRateMode,
    ServiceRunType,
    TimingMode,
    VideoFormat,
    VideoSynthType,
)
from aiperf.common.enums.dataset_enums import DatasetSamplingStrategy


#
# Config Defaults
@dataclass(frozen=True)
class CLIDefaults:
    TEMPLATE_FILENAME = "aiperf_config.yaml"


@dataclass(frozen=True)
class EndpointDefaults:
    MODEL_SELECTION_STRATEGY = ModelSelectionStrategy.ROUND_ROBIN
    CUSTOM_ENDPOINT = None
    TYPE = EndpointType.CHAT
    STREAMING = False
    URL = "localhost:8000"
    TIMEOUT = 600.0
    API_KEY = None
    USE_LEGACY_MAX_TOKENS = False


@dataclass(frozen=True)
class InputDefaults:
    BATCH_SIZE = 1
    EXTRA = []
    HEADERS = []
    FILE = None
    FIXED_SCHEDULE = False
    FIXED_SCHEDULE_AUTO_OFFSET = False
    FIXED_SCHEDULE_START_OFFSET = None
    FIXED_SCHEDULE_END_OFFSET = None
    GOODPUT = None
    PUBLIC_DATASET = None
    CUSTOM_DATASET_TYPE = None
    DATASET_SAMPLING_STRATEGY = DatasetSamplingStrategy.SHUFFLE
    RANDOM_SEED = None
    NUM_DATASET_ENTRIES = 100
    RANKINGS_PASSAGES_MEAN = 1
    RANKINGS_PASSAGES_STDDEV = 0


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
class VideoDefaults:
    BATCH_SIZE = 1
    DURATION = 5.0
    FPS = 4
    WIDTH = None
    HEIGHT = None
    SYNTH_TYPE = VideoSynthType.MOVING_SHAPES
    FORMAT = VideoFormat.WEBM
    CODEC = "libvpx-vp9"


@dataclass(frozen=True)
class PromptDefaults:
    BATCH_SIZE = 1
    NUM = 100


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
    NUM = None


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
    RAW_RECORDS_FOLDER = Path("raw_records")
    LOG_FOLDER = Path("logs")
    LOG_FILE = Path("aiperf.log")
    INPUTS_JSON_FILE = Path("inputs.json")
    PROFILE_EXPORT_AIPERF_CSV_FILE = Path("profile_export_aiperf.csv")
    PROFILE_EXPORT_AIPERF_JSON_FILE = Path("profile_export_aiperf.json")
    PROFILE_EXPORT_AIPERF_TIMESLICES_CSV_FILE = Path(
        "profile_export_aiperf_timeslices.csv"
    )
    PROFILE_EXPORT_AIPERF_TIMESLICES_JSON_FILE = Path(
        "profile_export_aiperf_timeslices.json"
    )
    PROFILE_EXPORT_JSONL_FILE = Path("profile_export.jsonl")
    PROFILE_EXPORT_RAW_JSONL_FILE = Path("profile_export_raw.jsonl")
    PROFILE_EXPORT_GPU_TELEMETRY_JSONL_FILE = Path("gpu_telemetry_export.jsonl")
    EXPORT_LEVEL = ExportLevel.RECORDS
    SLICE_DURATION = None


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
    LOG_LEVEL = AIPerfLogLevel.INFO
    VERBOSE = False
    EXTRA_VERBOSE = False
    LOG_PATH = None
    RECORD_PROCESSOR_SERVICE_COUNT = None
    UI_TYPE = AIPerfUIType.DASHBOARD


@dataclass(frozen=True)
class LoadGeneratorDefaults:
    BENCHMARK_DURATION = None
    BENCHMARK_GRACE_PERIOD = 30.0
    CONCURRENCY = None
    REQUEST_RATE = None
    REQUEST_COUNT = 10
    WARMUP_REQUEST_COUNT = 0
    REQUEST_RATE_MODE = RequestRateMode.POISSON
    TIMING_MODE = TimingMode.REQUEST_RATE
    REQUEST_CANCELLATION_RATE = 0.0
    REQUEST_CANCELLATION_DELAY = 0.0


@dataclass(frozen=True)
class WorkersDefaults:
    MIN = None
    MAX = None
