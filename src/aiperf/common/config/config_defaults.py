# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.enums import (
    AIPerfLogLevel,
    AudioFormat,
    ConnectionReuseStrategy,
    ExportLevel,
    ImageFormat,
    ModelSelectionStrategy,
    ServerMetricsFormat,
    VideoFormat,
    VideoSynthType,
)
from aiperf.plugin.enums import (
    ArrivalPattern,
    CommunicationBackend,
    DatasetSamplingStrategy,
    EndpointType,
    ServiceRunType,
    UIType,
    URLSelectionStrategy,
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
    TYPE = EndpointType.CHAT
    STREAMING = False
    URL = "localhost:8000"
    URL_STRATEGY = URLSelectionStrategy.ROUND_ROBIN
    TIMEOUT = 6 * 60 * 60  # 6 hours, match vLLM benchmark default
    API_KEY = None
    USE_LEGACY_MAX_TOKENS = False
    USE_SERVER_TOKEN_COUNT = False
    CONNECTION_REUSE_STRATEGY = ConnectionReuseStrategy.POOLED
    DOWNLOAD_VIDEO_CONTENT = False


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


@dataclass(frozen=True)
class RankingsDefaults:
    PASSAGES_MEAN = 1
    PASSAGES_STDDEV = 0
    PASSAGES_PROMPT_TOKEN_MEAN = 550
    PASSAGES_PROMPT_TOKEN_STDDEV = 0
    QUERY_PROMPT_TOKEN_MEAN = 550
    QUERY_PROMPT_TOKEN_STDDEV = 0


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
    SERVER_METRICS_EXPORT_JSONL_FILE = Path("server_metrics_export.jsonl")
    SERVER_METRICS_EXPORT_JSON_FILE = Path("server_metrics_export.json")
    SERVER_METRICS_EXPORT_CSV_FILE = Path("server_metrics_export.csv")
    SERVER_METRICS_EXPORT_PARQUET_FILE = Path("server_metrics_export.parquet")
    EXPORT_LEVEL = ExportLevel.RECORDS
    EXPORT_HTTP_TRACE = False
    SHOW_TRACE_TIMING = False
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
    UI_TYPE = UIType.DASHBOARD


@dataclass(frozen=True)
class LoadGeneratorDefaults:
    BENCHMARK_GRACE_PERIOD = 30.0
    MIN_REQUEST_COUNT = 10
    REQUEST_COUNT_MULTIPLIER = 2
    ARRIVAL_PATTERN = ArrivalPattern.POISSON


@dataclass(frozen=True)
class WorkersDefaults:
    MIN = None
    MAX = None


@dataclass(frozen=True)
class ServerMetricsDefaults:
    DEFAULT_FORMATS = [ServerMetricsFormat.JSON, ServerMetricsFormat.CSV]
