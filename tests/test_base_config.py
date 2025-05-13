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


import unittest

from copy import deepcopy
from enum import Enum
from unittest.mock import patch
from pathlib import PosixPath

from aiperf.common.models.config.config_field import ConfigField
from aiperf.common.models.config.base_config import BaseConfig
from aiperf.common.models.config.config_defaults import Range


class TestEnum(Enum):
    __test__ = False  # Prevents pytest from treating this as a test case
    OPTION_1 = "option_1"
    OPTION_2 = "option_2"
    OPTION_3 = "option_3"


class TestBaseConfig(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        pass

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    # Basic BaseConfig Testing
    ###########################################################################
    def test_base_config(self):
        """
        Test that a BaseConfig object can be written and read from
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )
        test_base_config.test_field_B = ConfigField(default=3, value=4)

        # Check that just the value is returned when accessing the attribute
        self.assertEqual(test_base_config.test_field_A, 2)
        self.assertEqual(test_base_config.test_field_B, 4)

        # Check the get_field() method
        self.assertEqual(test_base_config.get_field("test_field_A").value, 2)
        self.assertEqual(test_base_config.get_field("test_field_A").default, 1)
        self.assertEqual(
            test_base_config.get_field("test_field_A").template_comment, "test comment"
        )

        self.assertEqual(test_base_config.get_field("test_field_B").value, 4)

    def test_base_config_change_value(self):
        """
        Test that a BaseConfig object can have its values changed
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )

        # Change the value of a field
        test_base_config.test_field_A = 5

        # Check that the value has changed
        self.assertEqual(test_base_config.test_field_A, 5)

    def test_base_config_change_bounds(self):
        """
        Test that a BaseConfig object can have its bounds changed
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment", bounds={"upper": 3}
        )

        # Change the bounds of a field
        test_base_config.get_field("test_field_A").bounds = {"upper": 5}

        # Check that the value has changed
        self.assertEqual(
            test_base_config.get_field("test_field_A").bounds, {"upper": 5}
        )

    def test_base_config_change_choices_using_a_list(self):
        """
        Test that a BaseConfig object can have its choices changed using a list
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment", choices=[1, 2, 3]
        )

        # Change the choices of a field
        test_base_config.get_field("test_field_A").choices = [1, 2, 3, 4]

        # Check that the value has changed
        self.assertEqual(
            test_base_config.get_field("test_field_A").choices, [1, 2, 3, 4]
        )

    def test_base_config_change_choices_using_enum(self):
        """
        Test that a BaseConfig object can have its choices changed using an Enum
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=TestEnum.OPTION_1,
            value=TestEnum.OPTION_2,
            template_comment="test comment",
            choices=TestEnum,
        )

        self.assertEqual(test_base_config.get_field("test_field_A").choices, TestEnum)

        # Change the choices of a field
        test_base_config.get_field("test_field_A").choices = TestEnum.OPTION_3

        # Check that the value has changed
        self.assertEqual(
            test_base_config.get_field("test_field_A").choices, TestEnum.OPTION_3
        )

    def test_base_config_out_of_bounds_enum(self):
        """
        Test that a BaseConfig object with an out of bounds enum value raises an error
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=TestEnum.OPTION_1,
            value=TestEnum.OPTION_2,
            template_comment="test comment",
            choices=TestEnum,
        )

        # Change the value of a field to an out of bounds value
        with self.assertRaises(ValueError):
            test_base_config.test_field_A = "INVALID_VALUE"

    ###########################################################################
    # Utility Testing
    ###########################################################################
    def test_base_config_deepcopy(self):
        """
        Test that a BaseConfig object can be deepcopied
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )
        child_base_config = BaseConfig()
        child_base_config.test_field_B = ConfigField(default=3, value=4)
        test_base_config.child_element = child_base_config

        test_base_config_copy = deepcopy(test_base_config)

        # Check that the copied object is not the same object
        self.assertNotEqual(id(test_base_config), id(test_base_config_copy))

        # Check that the copied object is equal to the original object
        self.assertEqual(
            test_base_config.test_field_A, test_base_config_copy.test_field_A
        )

        # Modify the copied object
        test_base_config_copy.child_element.test_field_B = 6

        # Check that the copied object is not equal to the original object
        self.assertNotEqual(
            test_base_config.child_element.test_field_B,
            test_base_config_copy.child_element.test_field_B,
        )

    def test_model_dump(self):
        """
        Test that a BaseConfig object can be converted to a JSON schema
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )
        test_base_config.test_field_B = ConfigField(default=3, value=4)

        # Convert the object to a JSON schema
        test_schema = test_base_config.model_dump()

        # Check that the schema is correct
        self.assertEqual(test_schema["test_field_A"], 2)
        self.assertEqual(test_schema["test_field_B"], 4)

    def test_model_dump_nested(self):
        """
        Test that a BaseConfig object with nested objects can be converted to a JSON schema
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )
        test_base_config.test_field_B = BaseConfig()
        test_base_config.test_field_B.test_field_C = ConfigField(default=3, value=4)

        # Convert the object to a JSON schema
        test_schema = test_base_config.model_dump()

        # Check that the schema is correct
        self.assertEqual(test_schema["test_field_A"], 2)
        self.assertEqual(test_schema["test_field_B"]["test_field_C"], 4)

    def test_model_dump_enum(self):
        """
        Test that a BaseConfig object with an Enum can be converted to a JSON schema
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=TestEnum.OPTION_1,
            value=TestEnum.OPTION_2,
            template_comment="test comment",
            choices=TestEnum,
        )

        # Convert the object to a JSON schema
        test_schema = test_base_config.model_dump()

        # Check that the schema is correct
        self.assertEqual(test_schema["test_field_A"], "option_2")

    def test_model_dump_range(self):
        """
        Test that a BaseConfig object with a Range can be converted to a JSON schema
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=Range(min=0, max=100),
            value=Range(min=10, max=20),
            template_comment="test comment",
        )

        # Convert the object to a JSON schema
        test_schema = test_base_config.model_dump()

        # Check that the schema is correct
        self.assertEqual(test_schema["test_field_A"], {"min": 10, "max": 20})

    ###########################################################################
    # Template Testing
    ###########################################################################
    def test_template(self):
        """
        Test that a BaseConfig object can be converted to a template
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )
        test_base_config.test_field_B = ConfigField(default=3, value=4)

        # Create the template
        template = test_base_config.create_template(header="test")
        expected_template = (
            "  test:\n"
            + "    # test comment\n"
            + "    test_field_A: 2\n"
            + "    test_field_B: 4\n\n"
        )

        # Check that the template is correct
        self.assertEqual(
            template,
            expected_template,
        )

    ###########################################################################
    # Set by User Testing
    ###########################################################################
    def test_any_field_set_by_user_true(self):
        """
        Test that is_set_by_user returns True when at least one field is set by the user.
        """
        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(default=1, value=2)
        test_base_config.test_field_B = ConfigField(default=3)

        self.assertTrue(test_base_config.any_field_set_by_user())

    def test_any_field_set_by_user_false(self):
        """
        Test that is_set_by_user returns False when no fields are set by the user.
        """
        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(default=1)
        test_base_config.test_field_B = ConfigField(default=3)

        self.assertFalse(test_base_config.any_field_set_by_user())

    ###########################################################################
    # Check Required Fields Testing
    ###########################################################################
    def test_check_required_fields_all_set(self):
        """
        Test that check_required_fields_are_set does not raise an error when all required fields are set.
        """
        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(default=1, value=2, required=True)
        test_base_config.test_field_B = ConfigField(default=3, value=4, required=True)

        try:
            test_base_config.check_required_fields_are_set()
        except ValueError:
            self.fail("check_required_fields_are_set raised ValueError unexpectedly!")

    def test_check_required_fields_missing_field(self):
        """
        Test that check_required_fields_are_set raises an error when a required field is not set.
        """
        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(default=1, required=True)
        test_base_config.test_field_B = ConfigField(default=3, value=4, required=True)

        with self.assertRaises(ValueError) as context:
            test_base_config.check_required_fields_are_set()

        self.assertEqual(
            str(context.exception), "Required field test_field_A is not set"
        )

    def test_check_required_fields_are_set_nested(self):
        """
        Test that check_required_fields_are_set works correctly with nested BaseConfig objects.
        """
        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(default=1, value=2, required=True)

        child_base_config = BaseConfig()
        child_base_config.test_field_B = ConfigField(default=3, required=True)

        test_base_config.child_config = child_base_config

        with self.assertRaises(ValueError) as context:
            test_base_config.check_required_fields_are_set()

        self.assertEqual(
            str(context.exception), "Required field test_field_B is not set"
        )

    ###########################################################################
    # Get Legal JSON Value Testing
    ###########################################################################
    def test_get_legal_json_value_enum(self):
        """
        Test that _get_legal_json_value correctly handles Enum values.
        """
        test_base_config = BaseConfig()
        test_enum_value = TestEnum.OPTION_1
        result = test_base_config._get_legal_json_value(test_enum_value)
        self.assertEqual(result, "option_1")

    def test_get_legal_json_value_posix_path(self):
        """
        Test that _get_legal_json_value correctly handles PosixPath values.
        """
        test_base_config = BaseConfig()
        test_path = PosixPath("/test/path")
        result = test_base_config._get_legal_json_value(test_path)
        self.assertEqual(result, "/test/path")

    def test_get_legal_json_value_dict(self):
        """
        Test that _get_legal_json_value correctly handles dictionary values.
        """
        test_base_config = BaseConfig()
        test_dict = {"key1": TestEnum.OPTION_1, "key2": PosixPath("/test/path")}
        result = test_base_config._get_legal_json_value(test_dict)
        expected_result = {"key1": "option_1", "key2": "/test/path"}
        self.assertEqual(result, expected_result)

    def test_get_legal_json_value_object_with_dict(self):
        """
        Test that _get_legal_json_value correctly handles objects with __dict__.
        """

        class TestObject:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = 42

        test_base_config = BaseConfig()
        test_object = TestObject()
        result = test_base_config._get_legal_json_value(test_object)
        self.assertEqual(result, {"attr1": "value1", "attr2": 42})

    def test_get_legal_json_value_invalid_type(self):
        """
        Test that _get_legal_json_value raises a ValueError for unsupported types.
        """
        test_base_config = BaseConfig()
        with self.assertRaises(ValueError) as context:
            test_base_config._get_legal_json_value(set([1, 2, 3]))
        self.assertEqual(
            str(context.exception), "Value {1, 2, 3} is not a legal JSON value"
        )


if __name__ == "__main__":
    unittest.main()
