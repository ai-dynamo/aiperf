# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test calling from_json directly on non-base and leaf classes.

The hierarchy under test used to be the pub/sub command messages; it is defined
locally now that those are gone, because the behavior being pinned belongs to
``AutoRoutedModel``, not to any particular message.
"""

from tests.unit.common.models.test_auto_routed_messages import (
    Envelope,
    RequestEnvelope,
    SpawnRequest,
)


class TestDirectClassCalls:
    """Test behavior of from_json when called on different class levels."""

    def test_from_json_on_intermediate_class(self):
        """Calling from_json on RequestEnvelope (non-base, has discriminator)."""
        data = {
            "kind": "request",
            "action": "spawn",
            "num_workers": 5,
        }

        # Call from_json on RequestEnvelope directly (not Envelope)
        msg = RequestEnvelope.from_json(data)

        # RequestEnvelope has discriminator_field = "action"
        # So it WILL route to SpawnRequest
        assert isinstance(msg, SpawnRequest)
        assert msg.action == "spawn"
        assert msg.num_workers == 5

    def test_from_json_on_leaf_class(self):
        """Calling from_json on SpawnRequest (leaf, no discriminator)."""
        data = {
            "kind": "request",
            "action": "spawn",
            "num_workers": 10,
        }

        # Call from_json on leaf class directly
        msg = SpawnRequest.from_json(data)

        # SpawnRequest does NOT set discriminator_field
        # So it skips routing and validates directly as SpawnRequest
        assert isinstance(msg, SpawnRequest)
        assert msg.num_workers == 10

    def test_leaf_class_skips_validation_of_parent_discriminator(self):
        """Leaf class doesn't validate parent's discriminator value."""
        # This data has wrong message_type and command values
        # But SpawnRequest.from_json() will accept it!
        data = {
            "kind": "WRONG_KIND",  # Wrong value
            "action": "WRONG_ACTION",  # Wrong value
            "num_workers": 15,
        }

        # Leaf class skips routing, so it doesn't check discriminator values
        msg = SpawnRequest.from_json(data)

        # It just validates the data against the model fields
        assert isinstance(msg, SpawnRequest)
        assert msg.kind == "WRONG_KIND"  # Accepted as-is!
        assert msg.action == "WRONG_ACTION"  # Accepted as-is!
        assert msg.num_workers == 15

    def test_comparison_base_vs_intermediate_vs_leaf(self):
        """Compare behavior when calling from_json on different levels."""
        data = {
            "kind": "request",
            "action": "spawn",
            "num_workers": 20,
        }

        # All three produce the same result for valid data
        msg1 = Envelope.from_json(data)  # Routes: kind -> action
        msg2 = RequestEnvelope.from_json(data)  # Routes: action
        msg3 = SpawnRequest.from_json(data)  # No routing, direct validation

        assert isinstance(msg1, SpawnRequest)
        assert isinstance(msg2, SpawnRequest)
        assert isinstance(msg3, SpawnRequest)
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
            "kind": "not_checked",  # Not validated as a discriminator
            "action": "also_not_checked",  # Not validated as a discriminator
        }

        model = FlexibleModel.from_json(data)

        assert model.field1 == "hello"
        assert model.field2 == 99
        # Extra fields preserved due to extra="allow" in AIPerfBaseModel
        assert model.kind == "not_checked"
        assert model.action == "also_not_checked"
