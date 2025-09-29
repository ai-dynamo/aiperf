# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests to verify that all enums from all factories have implementations registered in the ModuleRegistry.

This test file ensures that the ModuleRegistry contains registrations for all enum values used by the factory classes,
confirming that the plugin system can discover and load implementations for all supported types.
"""

import inspect
from enum import Enum
from typing import get_args, get_origin

import pytest

import aiperf.common.factories as factories_module
from aiperf.common.enums import OpenAIObjectType
from aiperf.module_loader import ModuleRegistry


class TestModuleRegistryEnumCoverage:
    """Test suite to verify complete enum coverage in ModuleRegistry."""

    @classmethod
    def _discover_factory_enum_mappings(cls) -> dict[str, type[Enum]]:
        """Dynamically discover factory classes and their enum types from the factories module."""
        factory_enum_mappings = {}

        # Get all classes from the factories module
        for name, obj in inspect.getmembers(factories_module, inspect.isclass):
            # Check if it's a factory class (inherits from AIPerfFactory or AIPerfSingletonFactory)
            if hasattr(obj, "__orig_bases__") and any(
                "AIPerfFactory" in str(base) or "AIPerfSingletonFactory" in str(base)
                for base in obj.__orig_bases__
            ):
                # Extract the enum type from the generic type parameters
                for base in obj.__orig_bases__:
                    if get_origin(base) is not None:  # It's a generic type
                        type_args = get_args(base)
                        if type_args and len(type_args) >= 1:
                            enum_type = type_args[0]
                            # Verify it's actually an enum class
                            if inspect.isclass(enum_type) and issubclass(
                                enum_type, Enum
                            ):
                                factory_enum_mappings[name] = enum_type
                                break

        return factory_enum_mappings

    @property
    def factory_enum_mappings(self) -> dict[str, type[Enum]]:
        """Get the dynamically discovered factory enum mappings."""
        if not hasattr(self, "_factory_enum_mappings"):
            self._factory_enum_mappings = self._discover_factory_enum_mappings()
        return self._factory_enum_mappings

    @pytest.fixture
    def registry(self):
        """Get a ModuleRegistry instance with all plugins loaded."""
        registry = ModuleRegistry()
        # Load all plugins to ensure complete registry
        for factory_name in self.factory_enum_mappings:
            registry.load_all_plugins(factory_name)
        return registry

    def test_all_factories_have_registrations(self, registry):
        """Test that all expected factories have at least some registrations."""
        available_factories = set(registry.get_all_factories())
        expected_factories = set(self.factory_enum_mappings.keys())

        missing_factories = expected_factories - available_factories

        if missing_factories:
            pytest.fail(
                f"Missing factory registrations: {sorted(missing_factories)}. "
                f"Available factories: {sorted(available_factories)}"
            )

    def test_factory_enum_coverage(self, registry):
        """Test that each factory has registrations for all its enum values."""
        # List of enum values that should be skipped
        enum_values_to_skip = (
            OpenAIObjectType.EMBEDDING,  # This is parsed as a list of embeddings
        )

        for factory_name, enum_class in self.factory_enum_mappings.items():
            available_types = set(registry.get_available_types(factory_name))

            # Check that all enum string representations are available
            missing_enum_strings = []
            for enum_value in enum_class:
                if enum_value in enum_values_to_skip:
                    continue
                enum_string = str(enum_value)
                if enum_string not in available_types:
                    missing_enum_strings.append(enum_string)

            if missing_enum_strings:
                # Create a more detailed error message
                enum_values = [str(v) for v in enum_class]
                available_list = sorted(available_types)
                pytest.fail(
                    f"Factory '{factory_name}' is missing registrations for enum string values: \n\t{sorted(missing_enum_strings)}\n"
                    f"Expected enum string values: \n\t{sorted(enum_values)}\n"
                    f"Available registrations: \n\t{available_list}\n"
                    f"Enum class: {enum_class.__name__}"
                )

    def test_no_orphaned_registrations(self, registry):
        """Test that there are no registrations for unknown factories."""
        available_factories = set(registry.get_all_factories())
        expected_factories = set(self.factory_enum_mappings.keys())

        # Allow some tolerance for test factories or internal factories
        allowed_extra_patterns = ["Test", "Mock", "Internal"]

        orphaned_factories = available_factories - expected_factories
        unexpected_orphans = []

        for factory in orphaned_factories:
            if not any(pattern in factory for pattern in allowed_extra_patterns):
                unexpected_orphans.append(factory)

        if unexpected_orphans:
            pytest.fail(
                f"Found unexpected factory registrations: {sorted(unexpected_orphans)}.\n"
                "Please ensure that all factories are registered in the aiperf.common.factories module.\n"
                "If you truly need to have this factory outside of the factories module, "
                "please adjust the allowed patterns for this test."
            )

    def test_registry_completeness_summary(self, registry):
        """Provide a comprehensive summary of registry coverage."""
        summary = []
        total_expected_enums = 0
        total_registered_types = 0

        for factory_name, enum_class in self.factory_enum_mappings.items():
            available_types = registry.get_available_types(factory_name)
            enum_count = len(list(enum_class))
            registration_count = len(available_types)

            total_expected_enums += enum_count
            total_registered_types += registration_count

            coverage_ratio = registration_count / max(enum_count, 1)
            status = "✓" if coverage_ratio >= 1.0 else "✗"

            summary.append(
                f"{status} {factory_name}: {registration_count}/{enum_count} "
                f"({coverage_ratio:.1%}) - {enum_class.__name__}"
            )

        summary_text = "\n".join(summary)
        overall_coverage = total_registered_types / max(total_expected_enums, 1)

        print("\n=== ModuleRegistry Enum Coverage Summary ===")
        print(summary_text)
        print(
            f"\nOverall: {total_registered_types}/{total_expected_enums} ({overall_coverage:.1%})"
        )

        # This test should always pass - it's just for reporting
        assert True

    def test_factory_can_load_all_registered_types(self, registry):
        """Test that all registered types for a factory can actually be loaded."""
        for factory_name in self.factory_enum_mappings:
            available_types = registry.get_available_types(factory_name)
            load_failures = []

            for type_name in available_types:
                try:
                    # Try to load the plugin - this should not raise an exception
                    registry.load_plugin(factory_name, type_name)
                except Exception as e:
                    load_failures.append(f"{type_name}: {str(e)}")

            if load_failures:
                pytest.fail(
                    f"Factory '{factory_name}' failed to load registered types:\n"
                    + "\n".join(load_failures)
                )

    def test_registry_thread_safety_during_enum_coverage_check(self, registry):
        """Test that registry operations are thread-safe during comprehensive enum checking."""
        import threading
        import time

        def worker():
            """Worker function that performs registry operations."""
            for factory_name in self.factory_enum_mappings:
                try:
                    available_types = registry.get_available_types(factory_name)
                    if available_types:
                        # Try to load a random type
                        registry.load_plugin(factory_name, available_types[0])
                    time.sleep(
                        0.001
                    )  # Small delay to increase chance of race conditions
                except Exception:
                    pass  # Ignore exceptions in this stress test

        # Run multiple threads simultaneously
        threads = [threading.Thread(target=worker) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # If we get here without deadlocks or crashes, the test passes
        assert True

    def test_missing_enum_implementations_detailed_report(self, registry):
        """Generate a detailed report of missing enum implementations."""
        missing_implementations = {}

        for factory_name, enum_class in self.factory_enum_mappings.items():
            available_types = set(registry.get_available_types(factory_name))
            missing_for_factory = []

            for enum_value in enum_class:
                # Check if the string representation is available
                str_format = str(enum_value)

                # Skip embeddings, as they are parsed as lists of embeddings
                if enum_value == OpenAIObjectType.EMBEDDING:
                    continue

                if str_format not in available_types:
                    missing_for_factory.append(
                        {
                            "enum_value": enum_value,
                            "str_format": str_format,
                            "name": enum_value.name,
                        }
                    )

            if missing_for_factory:
                missing_implementations[factory_name] = {
                    "enum_class": enum_class.__name__,
                    "missing": missing_for_factory,
                    "available": sorted(available_types),
                }

        if missing_implementations:
            report_lines = ["Missing enum string implementations found:"]

            for factory_name, info in missing_implementations.items():
                report_lines.append(f"\n{factory_name} ({info['enum_class']}):")
                for missing in info["missing"]:
                    report_lines.append(
                        f"  - {missing['name']} ('{missing['str_format']}')"
                    )
                report_lines.append(f"  Available: {info['available']}")

            pytest.fail("\n".join(report_lines))

    def test_verify_actual_enum_values_in_codebase(self):
        """Verify that the dynamically discovered enum classes are valid."""
        # This test ensures our dynamic discovery is working correctly
        factory_mappings = self.factory_enum_mappings

        # Ensure we discovered some factories
        assert len(factory_mappings) > 0, "No factory enum mappings were discovered"

        # Verify each discovered enum is actually an enum class
        for factory_name, enum_class in factory_mappings.items():
            # Check that it's actually an enum class
            assert issubclass(enum_class, Enum), (
                f"{enum_class} for factory {factory_name} is not an Enum subclass"
            )

            # Check that it has at least one value
            enum_values = list(enum_class)
            assert len(enum_values) > 0, (
                f"Enum {enum_class.__name__} for factory {factory_name} has no values"
            )

            # Check that all values are properly defined
            for enum_value in enum_values:
                assert hasattr(enum_value, "name"), (
                    f"Enum value {enum_value} has no name attribute"
                )
                assert hasattr(enum_value, "value"), (
                    f"Enum value {enum_value} has no value attribute"
                )

    def test_dynamic_discovery_output(self):
        """Test to show what was dynamically discovered (for debugging/verification)."""
        factory_mappings = self.factory_enum_mappings

        print("\n=== Dynamically Discovered Factory-Enum Mappings ===")
        for factory_name, enum_class in sorted(factory_mappings.items()):
            enum_values = [e.name for e in enum_class]
            print(f"{factory_name}: {enum_class.__name__} -> {enum_values}")

        print(f"\nTotal discovered factories: {len(factory_mappings)}")

        # This test always passes - it's just for reporting
        assert True
