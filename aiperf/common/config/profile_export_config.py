# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for profile export file with dual-purpose parsing."""

from pathlib import Path

from pydantic import BaseModel

from aiperf.common.enums import ExportLevel


class ProfileExportConfig(BaseModel):
    """Configuration for profile export combining export level and file path."""

    export_level: ExportLevel
    file_path: Path


def parse_profile_export_file(
    value: str | ProfileExportConfig | None,
) -> ProfileExportConfig | None:
    """Parse --profile-export-file flag with dual-purpose syntax.

    Accepts:
    - Just enum: "records" -> level=records, path=default
    - Just path: "my_export.json" -> level=default, path=my_export.json
    - Enum:path: "records:my_export.json" -> level=records, path=my_export.json

    Args:
        value: Input string from CLI or ProfileExportConfig object

    Returns:
        ProfileExportConfig with export_level and file_path set
    """
    from aiperf.common.config.config_defaults import OutputDefaults

    if value is None:
        return ProfileExportConfig(
            export_level=ExportLevel(OutputDefaults.EXPORT_LEVEL),
            file_path=OutputDefaults.PROFILE_EXPORT_FILE,
        )
    if isinstance(value, ProfileExportConfig):
        return value

    if ":" in value:
        parts = value.split(":", maxsplit=1)
        level_str = parts[0].strip()
        path_str = parts[1].strip()

        try:
            export_level = ExportLevel(level_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid export level '{level_str}'. "
                f"Must be one of: {', '.join(e.value for e in ExportLevel)}"
            ) from e

        return ProfileExportConfig(
            export_level=export_level,
            file_path=Path(path_str),
        )

    # No colon - determine if it's an enum value or a path
    try:
        export_level = ExportLevel(value)
        # It's a valid enum, use default path
        return ProfileExportConfig(
            export_level=export_level,
            file_path=OutputDefaults.PROFILE_EXPORT_FILE,
        )
    except ValueError:
        # It's a path, use default level
        return ProfileExportConfig(
            export_level=ExportLevel(OutputDefaults.EXPORT_LEVEL),
            file_path=Path(value),
        )
