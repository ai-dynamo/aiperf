# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Braindead simple service runner using cyclopts for clean CLI argument handling."""

from cyclopts import App

from aiperf.cli_runner import run_service
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums.service_enums import ServiceType


def create_service_app(service_type: ServiceType) -> App:
    """Create a cyclopts app for a service."""
    app = App(
        name=f"aiperf-{service_type.replace('_', '-')}",
        help=f"Run {service_type} service",
    )

    @app.default
    def run(
        service_config: str,
        user_config: str,
        service_id: str | None = None,
        use_structured_logging: bool = False,
    ) -> None:
        """Run the service.

        Args:
            service_config: Service configuration as JSON string
            user_config: User configuration as JSON string
            service_id: Service ID (auto-generated if not provided)
            use_structured_logging: Enable structured logging for subprocess communication
        """
        try:
            service_config_obj = ServiceConfig.model_validate_json(service_config)
            user_config_obj = UserConfig.model_validate_json(user_config)

            # Auto-generate service_id if not provided
            if service_id is None:
                import uuid

                service_id = f"{service_type}_{uuid.uuid4().hex[:8]}"

            run_service(
                service_type,
                service_config_obj,
                user_config_obj,
                service_id=service_id,
                use_structured_subprocess_format=use_structured_logging,
            )
        except Exception as e:
            print(f"Error running service: {e}")
            raise SystemExit(1) from e

    return app
