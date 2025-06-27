# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass


# Temporary Record class to be used by the ConsoleExporter.
# TODO: Remove once the actual Records classes are fully implemented.
@dataclass
class Record:
    name: str
    unit: str
    avg: float
    min: float | None = None
    max: float | None = None
    p1: float | None = None
    p5: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    std: float | None = None
    streaming_only: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "name" and v is not None}
