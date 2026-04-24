# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.enums import (
    AIPerfLogLevel,
    ConnectionReuseStrategy,
    ExportLevel,
    ModelSelectionStrategy,
    RecordExportFormat,
)
from aiperf.plugin.enums import (
    CommunicationBackend,
    DatasetSamplingStrategy,
    EndpointType,
    ServiceRunType,
    UIType,
    URLSelectionStrategy,
)


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
    REQUEST_CONTENT_TYPE = None


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
class InputTokensDefaults:
    MEAN = 550
    STDDEV = 0.0
    BLOCK_SIZE = 512


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
    PROFILE_EXPORT_RECORDS_CSV_FILE = Path("profile_export_records.csv")
    PROFILE_EXPORT_JSONL_FILE = Path("profile_export.jsonl")
    PROFILE_EXPORT_RAW_JSONL_FILE = Path("profile_export_raw.jsonl")
    PROFILE_EXPORT_GPU_TELEMETRY_JSONL_FILE = Path("gpu_telemetry_export.jsonl")
    SERVER_METRICS_EXPORT_JSONL_FILE = Path("server_metrics_export.jsonl")
    SERVER_METRICS_EXPORT_JSON_FILE = Path("server_metrics_export.json")
    SERVER_METRICS_EXPORT_CSV_FILE = Path("server_metrics_export.csv")
    SERVER_METRICS_EXPORT_PARQUET_FILE = Path("server_metrics_export.parquet")
    EXPORT_LEVEL = ExportLevel.RECORDS
    EXPORT_HTTP_TRACE = False
    EXPORT_PER_CHUNK_DATA = False
    SHOW_TRACE_TIMING = False
    RECORD_EXPORT_FORMATS = [RecordExportFormat.JSONL]
    SLICE_DURATION = None


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
