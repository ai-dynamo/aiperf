# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

from aiperf.cli import app


def main() -> int:
    return app(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
