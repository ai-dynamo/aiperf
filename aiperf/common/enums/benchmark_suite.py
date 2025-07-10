# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base import CaseInsensitiveStrEnum


class BenchmarkSuiteCompletionTrigger(CaseInsensitiveStrEnum):
    """Determines how the suite completion is determined in order to know how to track the progress."""

    COMPLETED_PROFILES = "completed_profiles"
    """The suite will run until all profiles are completed."""

    # TODO: add other completion triggers
    # COMPLETED_SWEEPS = "completed_sweeps"
    # STABILIZATION_BASED = "stabilization_based"
    # CUSTOM = "custom"  # TBD


class BenchmarkSuiteType(CaseInsensitiveStrEnum):
    """Determines the type of suite to know how to track the progress."""

    SINGLE_PROFILE = "single_profile"
    """A suite with a single profile run."""

    # TODO: implement additional suite types
    # MULTI_PROFILE = "multi_profile"
    # """A suite with multiple profile runs. As opposed to a sweep, more than one parameter can be varied. TBD"""

    # SINGLE_SWEEP = "single_sweep"
    # """A suite with a single sweep over one or more varying parameters. TBD"""

    # MULTI_SWEEP = "multi_sweep"
    # """A suite with multiple sweep runs over multiple varying parameters. TBD"""

    # CUSTOM = "custom"
    # """User defined suite type. TBD"""
