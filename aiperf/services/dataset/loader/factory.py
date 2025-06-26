# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CustomDatasetType
from aiperf.common.factories import FactoryMixin
from aiperf.services.dataset.loader.protocol import CustomDatasetLoaderProtocol


class CustomDatasetFactory(
    FactoryMixin[CustomDatasetType, CustomDatasetLoaderProtocol]
):
    pass
