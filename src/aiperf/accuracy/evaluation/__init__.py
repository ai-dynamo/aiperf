# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral evaluator worker and Rust-owned host-effect boundary.

The modules in this package are deliberately independent of AIPerf's legacy
static/agentic evaluator implementations.  A selected evaluator provider owns
case semantics and aggregation, while :class:`PipeEvaluationHost` turns every
external effect into a typed protocol event for the supervising Rust runner.
"""

from aiperf.accuracy.evaluation.contracts import (
    CaseOutcome,
    CaseOutcomeKind,
    EvaluationBundle,
    EvaluationIdentity,
    EvaluationPlan,
    HostOperationRequest,
)
from aiperf.accuracy.evaluation.host import PipeEvaluationHost

__all__ = [
    "CaseOutcome",
    "CaseOutcomeKind",
    "EvaluationBundle",
    "EvaluationIdentity",
    "EvaluationPlan",
    "HostOperationRequest",
    "PipeEvaluationHost",
]
