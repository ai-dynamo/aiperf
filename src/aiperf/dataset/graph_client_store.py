# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client-store plugin face for agent graph runs.

Graph credits address the unified segment store by ``(trace_id,
node_ordinal)`` through the worker's ``GraphSegmentUnifiedClient`` -- there
are no conversations to serve. This class exists so the worker's generic
client-store construction (``plugins.get_class(DATASET_CLIENT_STORE,
client_metadata.client_type)``) has a registered target for
``GraphSegmentClientMetadata``; conversation lookups are an explicit hard
error, never a silent empty result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.exceptions import InvalidOperationError
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models.dataset_models import GraphSegmentClientMetadata

if TYPE_CHECKING:
    from aiperf.common.models.dataset_models import Conversation


class GraphSegmentDatasetClientStore(AIPerfLifecycleMixin):
    """No-op lifecycle wrapper carrying the graph store locations."""

    def __init__(self, client_metadata: GraphSegmentClientMetadata, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client_metadata = client_metadata

    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Graph runs have no conversation store; always raises."""
        raise InvalidOperationError(
            f"graph runs have no conversation store (requested "
            f"{conversation_id!r}); graph payloads are addressed by "
            "(trace_id, node_ordinal) in the unified segment store"
        )
