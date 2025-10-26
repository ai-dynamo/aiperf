# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test calling from_json directly on non-base and leaf classes."""

from aiperf.common.enums import CommandType
from aiperf.common.messages import Message
from aiperf.common.messages.command_messages import (
    CommandMessage,
    SpawnWorkersCommand,
)


class TestDirectClassCalls:
    """Test behavior of from_json when called on different class levels."""

    def test_from_json_on_intermediate_class(self):
        """Calling from_json on CommandMessage (non-base, has discriminator)."""
        data = {
            "message_type": "command",
            "command": "spawn_workers",
            "service_id": "controller",
            "num_workers": 5,
        }

        # Call from_json on CommandMessage directly (not Message)
        msg = CommandMessage.from_json(data)

        # CommandMessage has discriminator_field = "command"
        # So it WILL route to SpawnWorkersCommand
        assert isinstance(msg, SpawnWorkersCommand)
        assert msg.command == CommandType.SPAWN_WORKERS
        assert msg.num_workers == 5

    def test_from_json_on_leaf_class(self):
        """Calling from_json on SpawnWorkersCommand (leaf, no discriminator)."""
        data = {
            "message_type": "command",
            "command": "spawn_workers",
            "service_id": "controller",
            "num_workers": 10,
        }

        # Call from_json on leaf class directly
        msg = SpawnWorkersCommand.from_json(data)

        # SpawnWorkersCommand does NOT set discriminator_field
        # So it skips routing and validates directly as SpawnWorkersCommand
        assert isinstance(msg, SpawnWorkersCommand)
        assert msg.num_workers == 10

    def test_leaf_class_skips_validation_of_parent_discriminator(self):
        """Leaf class doesn't validate parent's discriminator value."""
        # This data has wrong message_type and command values
        # But SpawnWorkersCommand.from_json() will accept it!
        data = {
            "message_type": "WRONG_TYPE",  # Wrong value
            "command": "WRONG_COMMAND",  # Wrong value
            "service_id": "controller",
            "num_workers": 15,
        }

        # Leaf class skips routing, so it doesn't check discriminator values
        msg = SpawnWorkersCommand.from_json(data)

        # It just validates the data against the model fields
        assert isinstance(msg, SpawnWorkersCommand)
        assert msg.message_type == "WRONG_TYPE"  # Accepted as-is!
        assert msg.command == "WRONG_COMMAND"  # Accepted as-is!
        assert msg.num_workers == 15

    def test_comparison_base_vs_intermediate_vs_leaf(self):
        """Compare behavior when calling from_json on different levels."""
        data = {
            "message_type": "command",
            "command": "spawn_workers",
            "service_id": "controller",
            "num_workers": 20,
        }

        # All three produce the same result for valid data
        msg1 = Message.from_json(data)  # Routes: message_type -> command
        msg2 = CommandMessage.from_json(data)  # Routes: command
        msg3 = SpawnWorkersCommand.from_json(data)  # No routing, direct validation

        assert isinstance(msg1, SpawnWorkersCommand)
        assert isinstance(msg2, SpawnWorkersCommand)
        assert isinstance(msg3, SpawnWorkersCommand)
        assert msg1.num_workers == msg2.num_workers == msg3.num_workers == 20

    def test_model_with_no_discriminator_in_chain(self):
        """Model inheriting from AutoRoutedModel but with NO discriminator works like regular Pydantic."""
        from aiperf.common.models.base_models import AIPerfBaseModel

        # AIPerfBaseModel inherits from AutoRoutedModel but doesn't set discriminator_field
        # So it should work like a regular Pydantic model
        class SimpleModel(AIPerfBaseModel):
            name: str
            value: int

        data = {"name": "test", "value": 42}

        # from_json works, just validates directly (no routing)
        model = SimpleModel.from_json(data)

        assert isinstance(model, SimpleModel)
        assert model.name == "test"
        assert model.value == 42

    def test_model_with_no_discriminator_accepts_any_data(self):
        """Model without discriminator doesn't enforce any routing constraints."""
        from aiperf.common.models.base_models import AIPerfBaseModel

        class FlexibleModel(AIPerfBaseModel):
            field1: str
            field2: int

        # Can have fields that look like discriminators, doesn't matter
        data = {
            "field1": "hello",
            "field2": 99,
            "message_type": "not_checked",  # Not validated as a discriminator
            "command": "also_not_checked",  # Not validated as a discriminator
        }

        model = FlexibleModel.from_json(data)

        assert model.field1 == "hello"
        assert model.field2 == 99
        # Extra fields preserved due to extra="allow" in AIPerfBaseModel
        assert model.message_type == "not_checked"
        assert model.command == "also_not_checked"
