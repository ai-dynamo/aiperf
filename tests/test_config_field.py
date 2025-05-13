#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from enum import Enum

import pytest
from aiperf.common.models.config.config_field import ConfigField


def test_config_field_initialization():
    """
    Test the initialization of the ConfigField class.
    This test verifies that the ConfigField object is correctly initialized with the
    provided parameters and that its attributes have the expected default values.
    Assertions:
    - The `default` attribute is set to the provided value.
    - The `required` attribute is set to the provided value.
    - The `add_to_template` attribute is set to the provided value.
    - The `template_comment` attribute is set to the provided value.
    - The `value` attribute is None by default.
    - The `bounds` attribute is None by default.
    - The `choices` attribute is None by default.
    - The `is_set_by_user` attribute is False by default.
    """

    field = ConfigField(
        default=10,
        required=True,
        add_to_template=False,
        template_comment="Test comment",
    )
    assert field.default == 10
    assert field.required is True
    assert field.add_to_template is False
    assert field.template_comment == "Test comment"
    assert field.value is None
    assert field.bounds is None
    assert field.choices is None
    assert field.is_set_by_user is False


def test_config_field_set_value_within_bounds():
    """
    Test that the `ConfigField` class correctly sets the value within the specified bounds.

    This test verifies that when a value is assigned to the `ConfigField` instance,
    it respects the defined bounds and updates the value accordingly.

    Steps:
    1. Create a `ConfigField` instance with a default value of 5 and bounds of 0 (lower) and 10 (upper).
    2. Set the `value` attribute to 7, which is within the specified bounds.
    3. Assert that the `value` attribute is updated to 7.

    Expected Result:
    The `value` attribute of the `ConfigField` instance should be updated to 7 without any errors.
    """
    field = ConfigField(default=5, bounds={"lower": 0, "upper": 10})
    field.value = 7
    assert field.value == 7


def test_config_field_set_value_exceeds_upper_bound():
    """
    Test that setting a value exceeding the upper bound of a ConfigField raises a ValueError.

    This test verifies that when a value greater than the specified upper bound is assigned
    to the `value` attribute of a `ConfigField` instance, a `ValueError` is raised with the
    appropriate error message.

    Steps:
    1. Create a `ConfigField` instance with a default value of 5 and bounds of 0 (lower) and 10 (upper).
    2. Attempt to set the `value` attribute to 15, which exceeds the upper bound.
    3. Assert that a `ValueError` is raised with a message indicating the value exceeds the upper bounds.
    """
    field = ConfigField(default=5, bounds={"lower": 0, "upper": 10})
    with pytest.raises(ValueError, match="exceeds upper bounds"):
        field.value = 15


def test_config_field_set_value_below_lower_bound():
    """
    Test that setting a value below the lower bound of a ConfigField raises a ValueError.

    This test verifies that the ConfigField class enforces its lower bound constraint
    by raising a ValueError when an attempt is made to set its value below the defined
    lower bound.

    Steps:
    1. Create a ConfigField instance with a default value of 5 and bounds of 0 (lower) and 10 (upper).
    2. Attempt to set the field's value to -1, which is below the lower bound.
    3. Assert that a ValueError is raised with a message indicating the value exceeds the lower bounds.
    """
    field = ConfigField(default=5, bounds={"lower": 0, "upper": 10})
    with pytest.raises(ValueError, match="exceeds lower bounds"):
        field.value = -1


def test_config_field_set_invalid_bounds_key():
    """
    Test that a ValueError is raised when an invalid key is provided in the bounds dictionary
    of a ConfigField instance.

    This test ensures that the ConfigField class validates the keys in the bounds dictionary
    and raises an appropriate error if an invalid key is encountered.

    Expected Behavior:
    - A ValueError is raised with a message indicating that the provided key is not valid.

    Raises:
    - ValueError: If the bounds dictionary contains an invalid key.
    """
    with pytest.raises(ValueError, match="is not a valid key for bounds"):
        ConfigField(default=5, bounds={"invalid_key": 10})


def test_config_field_set_value_with_choices_list():
    """
    Test that the `ConfigField` class correctly sets and retrieves a value
    when the value is within the allowed choices.

    This test verifies:
    - The `ConfigField` instance is initialized with a default value and a list of valid choices.
    - The `value` attribute can be updated to a valid choice from the `choices` list.
    - The updated value is correctly stored and retrieved.

    Steps:
    1. Create a `ConfigField` instance with a default value of "A" and choices ["A", "B", "C"].
    2. Set the `value` attribute to "B".
    3. Assert that the `value` attribute is updated to "B".
    """
    field = ConfigField(default="A", choices=["A", "B", "C"])
    field.value = "B"
    assert field.value == "B"


def test_config_field_set_value_not_in_choices_list():
    """
    Test that setting a value not included in the list of valid choices for a ConfigField
    raises a ValueError with the appropriate error message.

    This test verifies:
    - A ConfigField instance is initialized with a default value and a list of valid choices.
    - Attempting to assign a value outside the defined choices raises a ValueError.
    - The error message explicitly mentions that the value is not in the list of choices.
    """
    field = ConfigField(default="A", choices=["A", "B", "C"])
    with pytest.raises(ValueError, match="not in list of choices"):
        field.value = "D"


def test_config_field_set_value_with_enum_choices():
    """
    Test the `set_value` functionality of the `ConfigField` class when using an enum for choices.

    This test ensures that:
    1. A `ConfigField` instance can be initialized with a default value from an enum.
    2. The `value` attribute of the `ConfigField` can be updated to another valid enum option.
    3. The updated value is correctly reflected and matches the expected enum option.

    Assertions:
    - Verify that the `value` attribute is updated to the new enum option.
    """

    class TestEnum(Enum):
        OPTION_A = "A"
        OPTION_B = "B"

    field = ConfigField(default=TestEnum.OPTION_A, choices=TestEnum)
    field.value = TestEnum.OPTION_B
    assert field.value == TestEnum.OPTION_B


def test_config_field_set_value_not_in_enum_choices():
    """
    Test that assigning a value not present in the enum choices to a ConfigField
    raises a ValueError.

    This test verifies that when attempting to set the `value` attribute of a
    `ConfigField` instance to a value that is not part of the specified `choices`
    (in this case, an enum), a `ValueError` is raised with the appropriate error
    message.

    Steps:
    1. Create a `ConfigField` instance with a default value and a set of enum choices.
    2. Attempt to assign an invalid value to the `value` attribute.
    3. Assert that a `ValueError` is raised with the expected error message.

    Expected Behavior:
    - A `ValueError` is raised with the message "not in list of choices" when
      an invalid value is assigned.
    """

    class TestEnum(Enum):
        OPTION_A = "A"
        OPTION_B = "B"

    field = ConfigField(default=TestEnum.OPTION_A, choices=TestEnum)
    with pytest.raises(ValueError, match="not in list of choices"):
        field.value = "INVALID_OPTION"


def test_config_field_is_set_by_user():
    """
    Test the behavior of the `is_set_by_user` property in the `ConfigField` class.

    This test verifies that:
    1. The `is_set_by_user` property is initially `False` when the field is set to its default value.
    2. The `is_set_by_user` property becomes `True` after the field's value is explicitly set by the user.
    """
    field = ConfigField(default=10)
    assert field.is_set_by_user is False
    field.value = 20
    assert field.is_set_by_user is True


def test_config_field_equality():
    """
    Test the equality and inequality behavior of the ConfigField class.

    This test verifies that two ConfigField instances with identical default
    values and bounds are considered equal. It also ensures that modifying
    the value of one instance causes the two instances to be considered
    unequal.

    Assertions:
        - Two ConfigField instances with the same default value and bounds
          are equal.
        - Modifying the value of one ConfigField instance makes it unequal
          to the other.
    """
    field1 = ConfigField(default=10, bounds={"lower": 0, "upper": 20})
    field2 = ConfigField(default=10, bounds={"lower": 0, "upper": 20})
    assert field1 == field2

    field2.value = 15
    assert field1 != field2
