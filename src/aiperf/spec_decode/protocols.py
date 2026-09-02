# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiperf.common.models import ParsedResponse, SpecDecodeAcceptanceRecord


@runtime_checkable
class SpecDecodeAdapterProtocol(Protocol):
    """Engine-specific adapter that fills the engine-neutral acceptance record.

    An adapter is the only component that knows an engine's on-the-wire
    spec-decode shape. It reads the raw payload captured on the parsed
    responses (``ParsedResponse.spec_decode_stats``) and produces a
    ``SpecDecodeAcceptanceRecord`` so nothing engine-specific leaks past it.

    Adapters are stateless and resolved by auto-detection: the parser walks
    every registered adapter in priority order and uses the first whose
    ``can_adapt`` returns True (see the ``spec_decode_adapter`` plugin
    category). Both methods are classmethods so no instance is constructed.
    """

    @classmethod
    def can_adapt(cls, responses: list[ParsedResponse]) -> bool:
        """Return True if these responses carry this engine's spec-decode payload.

        Must be cheap and side-effect free -- it runs on every record. Return
        False when no response carries a payload this adapter recognizes so the
        parser falls through to the next adapter (or produces no record).
        """
        ...

    @classmethod
    def adapt(
        cls, responses: list[ParsedResponse]
    ) -> SpecDecodeAcceptanceRecord | None:
        """Build the engine-neutral record from the raw payload.

        Returns None when the payload is absent or malformed. Only called when
        ``can_adapt`` returned True, but must still degrade to None rather than
        raise on an unexpected shape so a single bad record cannot abort a run.
        """
        ...
