# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Theme for Textual UI."""

from textual.theme import Theme

AIPERF_THEME = Theme(
    name="AIPerf Theme",
    primary="#76B900",
    secondary="#5A8A00",
    accent="#00D4FF",
    foreground="#E8E8E8",
    background="#1A1A1A",
    success="#76B900",
    warning="#FFA500",
    error="#FF4444",
    surface="#1A1A1A",
    panel="#252525",
    dark=True,
    variables={
        "block-cursor-text-style": "none",
        "footer-key-foreground": "#76B900",
        "input-selection-background": "#76B900 25%",
    },
)
