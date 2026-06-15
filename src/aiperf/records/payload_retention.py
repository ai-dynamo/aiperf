# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide whether a worker may drop canonical request ``payload_bytes`` from the
slim ``RecordContext`` after dispatch.

Stripping ``payload_bytes`` substantially reduces record-pipeline memory for
very large prompts, but the record pipeline has exactly three consumers that
read it:

1. Client-side input tokenization -- ``InferenceResultParser`` decodes the
   payload to count input (ISL) tokens.
2. Media counting from request bodies -- image / audio / video metrics derive
   their counts from the endpoint's single-pass ``extract_payload_inputs`` over
   the payload.
3. Raw payload export -- ``raw_record_writer_processor`` splices the bytes into
   the exported JSONL.

``record_payload_bytes_required`` returns True when *any* of those is active for
the run; a worker may strip the bytes whenever it returns False without changing
any observable metric. This module is the single source of truth tying the strip
decision to those consumers -- a new ``payload_bytes`` reader MUST be reflected
in ``record_payload_bytes_required`` (and guarded by
``tests/unit/records/test_payload_retention.py``), or auto-detection will
silently feed it ``None``.
"""

from __future__ import annotations

from aiperf.common.config import UserConfig
from aiperf.common.enums import ExportLevel
from aiperf.common.environment import Environment
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.plugin import plugins
from aiperf.plugin.schema.schemas import EndpointMetadata


def resolve_disable_tokenization(
    user_config: UserConfig, endpoint_meta: EndpointMetadata
) -> bool:
    """Whether client-side input tokenization is disabled for this run.

    Single source of truth shared with ``InferenceResultParser`` (whose ISL
    counting reads ``payload_bytes``): tokenization is off when the user
    requested server-reported counts, or the endpoint neither produces nor
    tokenizes tokens.

    Args:
        user_config: The resolved user configuration for the run.
        endpoint_meta: Plugin metadata for the run's endpoint type.

    Returns:
        True when no client-side tokenization will run.
    """
    return user_config.endpoint.use_server_token_count or (
        not endpoint_meta.produces_tokens and not endpoint_meta.tokenizes_input
    )


def _run_has_synthetic_media(user_config: UserConfig) -> bool:
    """Whether the run synthesizes image / audio / video inputs.

    Mirrors the inclusion predicates in ``SyntheticDatasetComposer``
    (``include_image`` / ``include_audio`` / ``include_video``). Note this does
    not detect media embedded in custom dataset payloads; see
    ``record_payload_bytes_required``.

    Args:
        user_config: The resolved user configuration for the run.

    Returns:
        True when any synthetic media modality is enabled.
    """
    media = user_config.input
    has_image = media.image.width.mean > 0 and media.image.height.mean > 0
    has_audio = media.audio.length.mean > 0
    has_video = bool(media.video.width and media.video.height)
    return has_image or has_audio or has_video


def record_payload_bytes_required(
    user_config: UserConfig, model_endpoint: ModelEndpointInfo
) -> bool:
    """True when a downstream record consumer reads ``payload_bytes``.

    The three consumers are client-side input tokenization, media counting from
    request bodies, and raw payload export. When all three are inert, a worker
    may drop ``payload_bytes`` from the record without changing any observable
    metric.

    Note: media embedded in *custom dataset* payloads (as opposed to synthetic
    ``--image/audio/video`` inputs) is not detected here. A run that relies on
    such media under server-token-count mode with non-raw export must retain
    payload bytes explicitly via ``AIPERF_RECORD_STRIP_PAYLOAD_BYTES=false``.

    Args:
        user_config: The resolved user configuration for the run.
        model_endpoint: The resolved endpoint info for the run.

    Returns:
        True when at least one consumer needs ``payload_bytes``.
    """
    endpoint_meta = plugins.get_endpoint_metadata(model_endpoint.endpoint.type)
    needs_client_isl = not resolve_disable_tokenization(user_config, endpoint_meta)
    needs_media_counts = _run_has_synthetic_media(user_config)
    needs_raw_export = user_config.output.export_level == ExportLevel.RAW
    return needs_client_isl or needs_media_counts or needs_raw_export


def resolve_strip_record_payload_bytes(
    user_config: UserConfig, model_endpoint: ModelEndpointInfo
) -> bool:
    """Resolve the tri-state ``AIPERF_RECORD_STRIP_PAYLOAD_BYTES`` to a bool.

    ``None`` (unset, the default) auto-detects: strip iff no downstream consumer
    needs the bytes (``not record_payload_bytes_required``). An explicit ``True``
    forces stripping (accepting the loss of client-side tokenization, media
    counting, and raw export); ``False`` always retains the bytes.

    Args:
        user_config: The resolved user configuration for the run.
        model_endpoint: The resolved endpoint info for the run.

    Returns:
        True when the worker should omit ``payload_bytes`` from records.
    """
    setting = Environment.RECORD.STRIP_PAYLOAD_BYTES
    if setting is not None:
        return setting
    return not record_payload_bytes_required(user_config, model_endpoint)
