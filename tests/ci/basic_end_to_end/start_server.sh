#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script now uses the main.py --all-servers approach
# instead of manually starting servers

set -x

echo "Starting end-to-end documentation tests using main.py --all-servers"

# Navigate to the test directory
cd "$(dirname "$0")"

# Execute the main.py script with --all-servers flag
python3 main.py --all-servers

echo "End-to-end documentation tests completed"