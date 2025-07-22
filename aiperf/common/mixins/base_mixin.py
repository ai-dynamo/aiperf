# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class BaseMixin:
    """Base mixin class.

    This Mixin creates a contract that Mixins should always pass **kwargs to
    super().__init__, regardless of whether they extend another mixin or not.

    This will ensure that the BaseMixin is the last mixin to have its __init__
    method called, which means that all other mixins will have a proper
    chain of __init__ methods with the correct arguments and no accidental
    broken inheritance.
    """

    def __init__(self, **kwargs):
        # object.__init__ does not take any arguments
        super().__init__()
