# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-specific transport support.

Everything here is service-agnostic: the eventstream reader dispatches on the
response content type, not on which AWS service produced it, so Bedrock's
streaming responses decode through the same path SageMaker's do.
"""
