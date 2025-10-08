# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Central registry of all factory registrations in AIPerf.

This file contains all Factory.register() calls with their corresponding
class paths, organized by factory type and sorted alphabetically within
each section.
"""

from aiperf.common.enums import (
    AIPerfUIType,
    CommClientType,
    CommunicationBackend,
    ComposerType,
    ConsoleExporterType,
    CustomDatasetType,
    DataExporterType,
    EndpointType,
    OpenAIObjectType,
    RecordProcessorType,
    RequestRateMode,
    ResultsProcessorType,
    ServiceRunType,
    ServiceType,
    TimingMode,
)
from aiperf.common.factories import (
    AIPerfUIFactory,
    CommunicationClientFactory,
    CommunicationFactory,
    ComposerFactory,
    ConsoleExporterFactory,
    CustomDatasetFactory,
    DataExporterFactory,
    InferenceClientFactory,
    OpenAIObjectParserFactory,
    RecordProcessorFactory,
    RequestConverterFactory,
    RequestRateGeneratorFactory,
    ResponseExtractorFactory,
    ResultsProcessorFactory,
    ServiceFactory,
    ServiceManagerFactory,
)
from aiperf.timing.credit_issuing_strategy import CreditIssuingStrategyFactory

# =============================================================================
# AIPerfUIFactory
# =============================================================================

AIPerfUIFactory.register(AIPerfUIType.DASHBOARD)(
    "aiperf.ui.dashboard.aiperf_dashboard_ui:AIPerfDashboardUI"
)
AIPerfUIFactory.register(AIPerfUIType.NONE)("aiperf.ui.no_ui:NoUI")
AIPerfUIFactory.register(AIPerfUIType.SIMPLE)("aiperf.ui.tqdm_ui:TQDMProgressUI")

# =============================================================================
# CommunicationClientFactory
# =============================================================================

CommunicationClientFactory.register(CommClientType.PUB)(
    "aiperf.zmq.pub_client:ZMQPubClient"
)
CommunicationClientFactory.register(CommClientType.PULL)(
    "aiperf.zmq.pull_client:ZMQPullClient"
)
CommunicationClientFactory.register(CommClientType.PUSH)(
    "aiperf.zmq.push_client:ZMQPushClient"
)
CommunicationClientFactory.register(CommClientType.REPLY)(
    "aiperf.zmq.router_reply_client:ZMQRouterReplyClient"
)
CommunicationClientFactory.register(CommClientType.REQUEST)(
    "aiperf.zmq.dealer_request_client:ZMQDealerRequestClient"
)
CommunicationClientFactory.register(CommClientType.SUB)(
    "aiperf.zmq.sub_client:ZMQSubClient"
)

# =============================================================================
# CommunicationFactory
# =============================================================================

CommunicationFactory.register(CommunicationBackend.ZMQ_IPC)(
    "aiperf.zmq.zmq_comms:ZMQIPCCommunication"
)
CommunicationFactory.register(CommunicationBackend.ZMQ_TCP)(
    "aiperf.zmq.zmq_comms:ZMQTCPCommunication"
)

# =============================================================================
# ComposerFactory
# =============================================================================

ComposerFactory.register(ComposerType.CUSTOM)(
    "aiperf.dataset.composer.custom:CustomDatasetComposer"
)
ComposerFactory.register(ComposerType.SYNTHETIC)(
    "aiperf.dataset.composer.synthetic:SyntheticDatasetComposer"
)

# =============================================================================
# ConsoleExporterFactory
# =============================================================================

ConsoleExporterFactory.register(ConsoleExporterType.ERRORS)(
    "aiperf.exporters.console_error_exporter:ConsoleErrorExporter"
)
ConsoleExporterFactory.register(ConsoleExporterType.EXPERIMENTAL_METRICS)(
    "aiperf.exporters.experimental_metrics_console_exporter:ConsoleExperimentalMetricsExporter"
)
ConsoleExporterFactory.register(ConsoleExporterType.INTERNAL_METRICS)(
    "aiperf.exporters.internal_metrics_console_exporter:ConsoleInternalMetricsExporter"
)
ConsoleExporterFactory.register(ConsoleExporterType.METRICS)(
    "aiperf.exporters.console_metrics_exporter:ConsoleMetricsExporter"
)

# =============================================================================
# CreditIssuingStrategyFactory
# =============================================================================

CreditIssuingStrategyFactory.register(TimingMode.FIXED_SCHEDULE)(
    "aiperf.timing.fixed_schedule_strategy:FixedScheduleStrategy"
)
CreditIssuingStrategyFactory.register(TimingMode.REQUEST_RATE)(
    "aiperf.timing.request_rate_strategy:RequestRateStrategy"
)

# =============================================================================
# CustomDatasetFactory
# =============================================================================

CustomDatasetFactory.register(CustomDatasetType.MOONCAKE_TRACE)(
    "aiperf.dataset.loader.mooncake_trace:MooncakeTraceDatasetLoader"
)
CustomDatasetFactory.register(CustomDatasetType.MULTI_TURN)(
    "aiperf.dataset.loader.multi_turn:MultiTurnDatasetLoader"
)
CustomDatasetFactory.register(CustomDatasetType.RANDOM_POOL)(
    "aiperf.dataset.loader.random_pool:RandomPoolDatasetLoader"
)
CustomDatasetFactory.register(CustomDatasetType.SINGLE_TURN)(
    "aiperf.dataset.loader.single_turn:SingleTurnDatasetLoader"
)

# =============================================================================
# DataExporterFactory
# =============================================================================

DataExporterFactory.register(DataExporterType.CSV)(
    "aiperf.exporters.csv_exporter:CsvExporter"
)
DataExporterFactory.register(DataExporterType.JSON)(
    "aiperf.exporters.json_exporter:JsonExporter"
)

# =============================================================================
# InferenceClientFactory
# =============================================================================

InferenceClientFactory.register_all(
    EndpointType.CHAT,
    EndpointType.COMPLETIONS,
    EndpointType.EMBEDDINGS,
    EndpointType.RANKINGS,
    EndpointType.RESPONSES,
)("aiperf.clients.openai.openai_aiohttp:OpenAIClientAioHttp")

# =============================================================================
# OpenAIObjectParserFactory
# =============================================================================

OpenAIObjectParserFactory.register(OpenAIObjectType.CHAT_COMPLETION)(
    "aiperf.parsers.openai_parsers:ChatCompletionParser"
)
OpenAIObjectParserFactory.register(OpenAIObjectType.CHAT_COMPLETION_CHUNK)(
    "aiperf.parsers.openai_parsers:ChatCompletionChunkParser"
)
OpenAIObjectParserFactory.register(OpenAIObjectType.COMPLETION)(
    "aiperf.parsers.openai_parsers:CompletionParser"
)
OpenAIObjectParserFactory.register(OpenAIObjectType.LIST)(
    "aiperf.parsers.openai_parsers:ListParser"
)
OpenAIObjectParserFactory.register(OpenAIObjectType.RANKINGS)(
    "aiperf.parsers.openai_parsers:RankingsParser"
)
OpenAIObjectParserFactory.register(OpenAIObjectType.RESPONSE)(
    "aiperf.parsers.openai_parsers:ResponseParser"
)
OpenAIObjectParserFactory.register(OpenAIObjectType.TEXT_COMPLETION)(
    "aiperf.parsers.openai_parsers:TextCompletionParser"
)

# =============================================================================
# RecordProcessorFactory
# =============================================================================

RecordProcessorFactory.register(RecordProcessorType.METRIC_RECORD)(
    "aiperf.post_processors.metric_record_processor:MetricRecordProcessor"
)

# =============================================================================
# RequestConverterFactory
# =============================================================================

RequestConverterFactory.register(EndpointType.CHAT)(
    "aiperf.clients.openai.openai_chat:OpenAIChatCompletionRequestConverter"
)
RequestConverterFactory.register(EndpointType.COMPLETIONS)(
    "aiperf.clients.openai.openai_completions:OpenAICompletionRequestConverter"
)
RequestConverterFactory.register(EndpointType.EMBEDDINGS)(
    "aiperf.clients.openai.openai_embeddings:OpenAIEmbeddingsRequestConverter"
)
RequestConverterFactory.register(EndpointType.RANKINGS)(
    "aiperf.clients.openai.rankings:RankingsRequestConverter"
)
RequestConverterFactory.register(EndpointType.RESPONSES)(
    "aiperf.clients.openai.openai_responses:OpenAIResponsesRequestConverter"
)

# =============================================================================
# RequestRateGeneratorFactory
# =============================================================================

RequestRateGeneratorFactory.register(RequestRateMode.CONCURRENCY_BURST)(
    "aiperf.timing.request_rate_strategy:ConcurrencyBurstRateGenerator"
)
RequestRateGeneratorFactory.register(RequestRateMode.CONSTANT)(
    "aiperf.timing.request_rate_strategy:ConstantRateGenerator"
)
RequestRateGeneratorFactory.register(RequestRateMode.POISSON)(
    "aiperf.timing.request_rate_strategy:PoissonRateGenerator"
)

# =============================================================================
# ResponseExtractorFactory
# =============================================================================

ResponseExtractorFactory.register_all(
    EndpointType.CHAT,
    EndpointType.COMPLETIONS,
    EndpointType.EMBEDDINGS,
    EndpointType.RANKINGS,
    EndpointType.RESPONSES,
)("aiperf.parsers.openai_parsers:OpenAIResponseExtractor")

# =============================================================================
# ResultsProcessorFactory
# =============================================================================

ResultsProcessorFactory.register(ResultsProcessorType.METRIC_RESULTS)(
    "aiperf.post_processors.metric_results_processor:MetricResultsProcessor"
)

# =============================================================================
# ServiceFactory
# =============================================================================

ServiceFactory.register(ServiceType.DATASET_MANAGER)(
    "aiperf.dataset.dataset_manager:DatasetManager"
)
ServiceFactory.register(ServiceType.RECORD_PROCESSOR)(
    "aiperf.records.record_processor_service:RecordProcessor"
)
ServiceFactory.register(ServiceType.RECORDS_MANAGER)(
    "aiperf.records.records_manager:RecordsManager"
)
ServiceFactory.register(ServiceType.SYSTEM_CONTROLLER)(
    "aiperf.controller.system_controller:SystemController"
)
ServiceFactory.register(ServiceType.TIMING_MANAGER)(
    "aiperf.timing.timing_manager:TimingManager"
)
ServiceFactory.register(ServiceType.WORKER)("aiperf.workers.worker:Worker")
ServiceFactory.register(ServiceType.WORKER_MANAGER)(
    "aiperf.workers.worker_manager:WorkerManager"
)

# =============================================================================
# ServiceManagerFactory
# =============================================================================

ServiceManagerFactory.register(ServiceRunType.KUBERNETES)(
    "aiperf.controller.kubernetes_service_manager:KubernetesServiceManager"
)
ServiceManagerFactory.register(ServiceRunType.MULTIPROCESSING)(
    "aiperf.controller.multiprocess_service_manager:MultiProcessServiceManager"
)
