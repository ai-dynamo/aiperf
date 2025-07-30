# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from pydantic.fields import FieldInfo

from aiperf.common.config import BaseConfig
from aiperf.common.models import AIPerfBaseModel


class NestedConfig(AIPerfBaseModel):
    field1: str
    field2: int


class BaseTestConfig(BaseConfig):
    nested: NestedConfig
    verbose: bool = False


def test_serialize_to_yaml():
    """
    Tests the `serialize_to_yaml` method of the `BaseTestConfig` class.

    This test verifies that the YAML serialization of a `BaseTestConfig` object
    correctly includes all nested fields and their values, as well as other
    attributes of the configuration.

    Assertions:
        - The serialized YAML output contains the "nested" key.
        - The serialized YAML output includes the "field1" and "field2" values
          from the `NestedConfig` object.
        - The serialized YAML output includes the "verbose" attribute with the
          correct value.
    """
    config = BaseTestConfig(
        nested=NestedConfig(field1="value1", field2=42),
        verbose=True,
    )

    yaml_output = config.serialize_to_yaml(verbose=config.verbose, indent=2)

    assert "nested:" in yaml_output
    assert "field1: value1" in yaml_output
    assert "field2: 42" in yaml_output
    assert "verbose: true" in yaml_output


def test_attach_comments_with_verbose():
    """
    Test the `serialize_to_yaml` method of `BaseTestConfig` when the `verbose` flag is set to `True`.

    This test ensures that:
    - Comments (e.g., descriptions) are not attached to the YAML output when `verbose` is `True`.
    - The serialized YAML output contains the expected structure, such as the presence of the "nested" key.

    Assertions:
    - Verify that the description comment for `field1` is not present in the YAML output.
    - Verify that the "nested:" key is included in the YAML output.
    """
    config = BaseTestConfig(
        nested=NestedConfig(field1="value1", field2=42),
        verbose=True,
    )

    yaml_output = config.serialize_to_yaml(verbose=config.verbose, indent=2)

    # Check if comments are attached when verbose is True
    assert (
        "# field1 description" not in yaml_output
    )  # No description provided in this example
    assert "nested:" in yaml_output


def test_preprocess_value_enum():
    """
    Test the `_preprocess_value` method of the `BaseConfig` class when handling
    an enumeration value.

    This test defines a sample enumeration `SampleEnum` with two options,
    `OPTION_A` and `OPTION_B`. It verifies that when an enumeration value
    (`SampleEnum.OPTION_A`) is passed to `_preprocess_value`, the method
    correctly processes it into a lowercase string representation of the
    enumeration value's name.

    Assertions:
        - The processed value of `SampleEnum.OPTION_A` should be "optiona".
    """

    class SampleEnum(Enum):
        OPTION_A = "OptionA"
        OPTION_B = "OptionB"

    value = SampleEnum.OPTION_A
    processed_value = BaseConfig._preprocess_value(value)

    assert processed_value == "optiona"


def test_should_add_field_to_template():
    """
    Test the `_should_add_field_to_template` method of the `BaseConfig` class.

    This test verifies the behavior of `_should_add_field_to_template` when handling
    fields with different `json_schema_extra` configurations.

    Assertions:
        - Fields with `add_to_template` set to `False` should not be added to the template.
        - Fields with `add_to_template` set to `True` should be added to the template.
        - Fields without `json_schema_extra` should be added to the template by default.
        - Fields with `json_schema_extra` set to `None` should be added to the template by default.
        - Fields with unexpected keys in `json_schema_extra` should be added to the template by default.
    """

    field_with_flag = FieldInfo(json_schema_extra={"add_to_template": False})
    field_with_add_to_template_true = FieldInfo(
        json_schema_extra={"add_to_template": True}
    )
    field_no_extra = FieldInfo()
    field_with_none_extra = FieldInfo(json_schema_extra=None)
    field_with_unexpected_keys = FieldInfo(json_schema_extra={"unexpected_key": True})

    assert not BaseConfig._should_add_field_to_template(field_with_flag)
    assert BaseConfig._should_add_field_to_template(field_with_add_to_template_true)
    assert BaseConfig._should_add_field_to_template(field_no_extra)
    assert BaseConfig._should_add_field_to_template(field_with_none_extra)
    assert BaseConfig._should_add_field_to_template(field_with_unexpected_keys)


def test_is_a_nested_config():
    """
    Test the `_is_a_nested_config` method of the `BaseConfig` class.

    This test verifies the behavior of the `_is_a_nested_config` method when:
    1. A valid nested configuration object is passed.
    2. An invalid input (not a dictionary) is passed.

    Assertions:
    - The method should return `True` when a valid nested configuration object is passed.
    - The method should return `False` when an invalid input is passed.
    """
    nested_model = NestedConfig(field1="value1", field2=42)
    field = BaseTestConfig.model_fields.get("nested")

    assert BaseConfig._is_a_nested_config(field, nested_model.model_dump())
    assert not BaseConfig._is_a_nested_config(field, "not a dict")
