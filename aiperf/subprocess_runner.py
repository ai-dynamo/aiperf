#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Subprocess runner for executing AIPerf services as separate processes.
This module provides the entry point for running services via asyncio.create_subprocess_exec.
"""

import argparse
import json
import sys

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig


def main():
    """Main entry point for subprocess service execution."""
    parser = argparse.ArgumentParser(description="AIPerf Service Subprocess Runner")
    parser.add_argument("--service-type", required=True, help="Type of service to run")
    parser.add_argument(
        "--service-id", required=True, help="Unique ID for the service instance"
    )
    parser.add_argument(
        "--service-config", required=True, help="JSON serialized service configuration"
    )
    parser.add_argument(
        "--user-config", required=True, help="JSON serialized user configuration"
    )
    parser.add_argument(
        "--use-structured-logging",
        action="store_true",
        help="Use structured logging format for subprocess parsing",
    )

    args = parser.parse_args()

    try:
        # Deserialize configurations
        try:
            service_config_dict = json.loads(args.service_config)
            user_config_dict = json.loads(args.user_config)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON configuration: {e}", file=sys.stderr)
            sys.exit(1)

        # Reconstruct config objects
        try:
            service_config = ServiceConfig.model_validate(service_config_dict)
            user_config = UserConfig.model_validate(user_config_dict)
        except Exception as e:
            print(f"Error validating configuration objects: {e}", file=sys.stderr)
            sys.exit(1)

        from aiperf.module_loader import ensure_modules_loaded

        ensure_modules_loaded()

        # Get service class from factory
        try:
            from aiperf.common.enums import ServiceType
            from aiperf.common.factories import ServiceFactory

            # Convert string to ServiceType enum
            service_type_enum = ServiceType(args.service_type)
            service_class = ServiceFactory.get_class_from_type(service_type_enum)
        except Exception as e:
            print(
                f"Error getting service class for type {args.service_type}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Bootstrap and run the service
        bootstrap_and_run_service(
            service_class=service_class,
            service_id=args.service_id,
            service_config=service_config,
            user_config=user_config,
            use_structured_subprocess_format=args.use_structured_logging,
        )

    except KeyboardInterrupt:
        print(
            f"Service {args.service_type} ({args.service_id}) interrupted",
            file=sys.stderr,
        )
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(
            f"Unexpected error running service {args.service_type} ({args.service_id}): {e}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
