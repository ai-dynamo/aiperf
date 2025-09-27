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
        service_id: str | None = None,
        service_config: str | None = None,
        user_config: str | None = None,
        use_structured_logging: bool = False,
    ) -> None:
        """Run the service.

        Args:
            service_id: Service ID for subprocess mode
            service_config: Service configuration as JSON string
            user_config: User configuration as JSON string
            use_structured_logging: Enable structured logging for subprocess communication
        """
        if service_config and user_config:
            # Subprocess mode: parse JSON configs and use cli_runner
            try:
                service_config_obj = ServiceConfig.model_validate_json(service_config)
                user_config_obj = UserConfig.model_validate_json(user_config)

                run_service(
                    service_type,
                    service_config_obj,
                    user_config_obj,
                    service_id=service_id,
                    use_structured_subprocess_format=use_structured_logging,
                )
            except Exception as e:
                print(f"Error in subprocess mode: {e}")
                raise SystemExit(1) from e
        else:
            # Standalone mode: use bootstrap_and_run_service
            from aiperf.common.bootstrap import bootstrap_and_run_service
            from aiperf.common.factories import ServiceFactory

            try:
                service_class = ServiceFactory.get_class_from_type(service_type)
                bootstrap_and_run_service(service_class)
            except Exception as e:
                print(f"Error in standalone mode: {e}")
                raise SystemExit(1) from e

    return app
