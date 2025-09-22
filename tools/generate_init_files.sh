#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

VENV_PATH="${VENV_PATH:-.venv}"
ACTIVATE_VENV="${VENV_PATH}/bin/activate"
if [ -f "${ACTIVATE_VENV}" ]; then
    source "${ACTIVATE_VENV}"
fi

mkinit --write --black --nomods --recursive aiperf
# Ruff check and fix just the __init__.py files, because mkinit sometimes
# doesn't sort the imports correctly, causing infinite error loops.
find aiperf -name '__init__.py' | xargs ruff check --fix
