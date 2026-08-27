# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical names for order-independent AIPerf random streams.

Ported verbatim from ``rust/aiperf/src/rng/namespace.rs``. These strings are part of the
reproducibility contract: changing a value reseeds that component for every deterministic
run. Kept identical to the Rust constants so a stream derived on either side matches.
"""

from __future__ import annotations

COMPOSER_CONVERSATION_TURN_COUNT = "composer.conversation.turn_count"
COMPOSER_CONVERSATION_TURN_DELAY = "composer.conversation.turn_delay"
COMPOSER_TURN_MAX_TOKENS = "composer.turn.max_tokens"
COMPOSER_TURN_MODEL_SELECTION = "composer.turn.model_selection"
DATASET_AUDIO_DATA = "dataset.audio.data"
DATASET_AUDIO_DURATION = "dataset.audio.duration"
DATASET_AUDIO_FORMAT = "dataset.audio.format"
DATASET_CODING_CONTENT_CORPUS = "dataset.coding_content.corpus"
DATASET_CODING_CONTENT_LENGTH = "dataset.coding_content.length"
DATASET_CODING_CONTENT_TEMPLATE = "dataset.coding_content.template"
DATASET_IMAGE_DIMENSIONS = "dataset.image.dimensions"
DATASET_IMAGE_FORMAT = "dataset.image.format"
DATASET_IMAGE_NOISE = "dataset.image.noise"
DATASET_IMAGE_SOURCE = "dataset.image.source"
DATASET_LOADER = "dataset.loader"
DATASET_LOADER_RANDOM_POOL = "dataset.loader.random_pool"
DATASET_LOADER_RANDOM_POOL_SAMPLING = "dataset.loader.random_pool.sampling"
DATASET_LOADER_SHAREGPT = "dataset.loader.sharegpt"
DATASET_PROMPT_CORPUS = "dataset.prompt.corpus"
DATASET_PROMPT_LENGTH = "dataset.prompt.length"
DATASET_PROMPT_PREFIX = "dataset.prompt.prefix"
DATASET_RANKINGS_PASSAGES = "dataset.rankings.passages"
DATASET_RANKINGS_PASSAGES_TOKENS = "dataset.rankings.passages.tokens"
DATASET_RANKINGS_QUERY_TOKENS = "dataset.rankings.query.tokens"
DATASET_SAMPLER_RANDOM = "dataset.sampler.random"
DATASET_SAMPLER_SHUFFLE = "dataset.sampler.shuffle"
DATASET_SYNTHESIS_EMPIRICAL_SAMPLER = "dataset.synthesis.empirical_sampler"
DATASET_SYNTHESIS_SYNTHESIZER = "dataset.synthesis.synthesizer"
DATASET_VIDEO_AUDIO = "dataset.video.audio"
DATASET_VIDEO_NOISE = "dataset.video.noise"
GRAPH_ARRIVAL = "graph.arrival"
GRAPH_NODE_CANCELLATION = "graph.node.cancellation"
GRAPH_NODE_CANCELLATION_WORKER = "graph.node.cancellation.worker"
GRAPH_PHASE = "graph.phase"
MOCK_DCGM = "mock.dcgm"
MOCK_ERRORS = "mock.errors"
MODELS_SEQUENCE_DISTRIBUTION = "models.sequence.distribution"
TIMING_RAMP_CONCURRENCY = "timing.ramp.concurrency"
TIMING_RAMP_POISSON = "timing.ramp.poisson"
TIMING_RAMP_PREFILL_CONCURRENCY = "timing.ramp.prefill_concurrency"
TIMING_RAMP_REQUEST_RATE = "timing.ramp.request_rate"
TIMING_REQUEST_CANCELLATION = "timing.request.cancellation"
TIMING_REQUEST_GAMMA_INTERVAL = "timing.request.gamma_interval"
TIMING_REQUEST_POISSON_INTERVAL = "timing.request.poisson_interval"
TIMING_REQUEST_RATE = "timing.request_rate"

# Every canonical stream name, in lexical order (``namespace.rs`` ``ALL``).
ALL: tuple[str, ...] = (
    COMPOSER_CONVERSATION_TURN_COUNT,
    COMPOSER_CONVERSATION_TURN_DELAY,
    COMPOSER_TURN_MAX_TOKENS,
    COMPOSER_TURN_MODEL_SELECTION,
    DATASET_AUDIO_DATA,
    DATASET_AUDIO_DURATION,
    DATASET_AUDIO_FORMAT,
    DATASET_CODING_CONTENT_CORPUS,
    DATASET_CODING_CONTENT_LENGTH,
    DATASET_CODING_CONTENT_TEMPLATE,
    DATASET_IMAGE_DIMENSIONS,
    DATASET_IMAGE_FORMAT,
    DATASET_IMAGE_NOISE,
    DATASET_IMAGE_SOURCE,
    DATASET_LOADER,
    DATASET_LOADER_RANDOM_POOL,
    DATASET_LOADER_RANDOM_POOL_SAMPLING,
    DATASET_LOADER_SHAREGPT,
    DATASET_PROMPT_CORPUS,
    DATASET_PROMPT_LENGTH,
    DATASET_PROMPT_PREFIX,
    DATASET_RANKINGS_PASSAGES,
    DATASET_RANKINGS_PASSAGES_TOKENS,
    DATASET_RANKINGS_QUERY_TOKENS,
    DATASET_SAMPLER_RANDOM,
    DATASET_SAMPLER_SHUFFLE,
    DATASET_SYNTHESIS_EMPIRICAL_SAMPLER,
    DATASET_SYNTHESIS_SYNTHESIZER,
    DATASET_VIDEO_AUDIO,
    DATASET_VIDEO_NOISE,
    GRAPH_ARRIVAL,
    GRAPH_NODE_CANCELLATION,
    GRAPH_NODE_CANCELLATION_WORKER,
    GRAPH_PHASE,
    MOCK_DCGM,
    MOCK_ERRORS,
    MODELS_SEQUENCE_DISTRIBUTION,
    TIMING_RAMP_CONCURRENCY,
    TIMING_RAMP_POISSON,
    TIMING_RAMP_PREFILL_CONCURRENCY,
    TIMING_RAMP_REQUEST_RATE,
    TIMING_REQUEST_CANCELLATION,
    TIMING_REQUEST_GAMMA_INTERVAL,
    TIMING_REQUEST_POISSON_INTERVAL,
    TIMING_REQUEST_RATE,
)
