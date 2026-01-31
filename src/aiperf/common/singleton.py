# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from abc import ABCMeta
from threading import Lock


class SingletonMeta(ABCMeta):
    """Metaclass that creates a Singleton instance per process, compatible with ABCMeta.

    This metaclass ensures that only one instance of a class can exist per process.
    It extends ABCMeta to maintain compatibility with abstract base classes.
    Each process gets its own singleton instance, making it safe for multiprocessing.
    """

    _instances = {}
    _instances_locks: dict[tuple[type, int], Lock] = {}
    # Keep a lock for the instances_locks dictionary to prevent race conditions
    _locks_lock = Lock()

    def __call__(cls, *args, **kwargs):
        """Override the call method to implement singleton behavior per process.

        Returns the existing instance if it exists in the current process,
        otherwise creates a new one.
        """
        # Use process ID as part of the key to ensure each process gets its own singleton
        key = (cls, os.getpid())
        if key not in cls._instances:
            # Get or create the lock for the instances_locks dictionary
            with cls._locks_lock:
                if key not in cls._instances_locks:
                    cls._instances_locks[key] = Lock()

            with cls._instances_locks[key]:
                # Double check after acquiring the lock to avoid race conditions
                if key not in cls._instances:
                    # This will call the __new__ method and then the __init__ method of the class
                    cls._instances[key] = super().__call__(*args, **kwargs)

        return cls._instances[key]


class Singleton(metaclass=SingletonMeta):
    """Base class that transforms any subclass into a per-process Singleton.

    Ensures that only one instance of the class can exist per process.
    Subsequent calls to create an instance within the same process will return
    the existing instance. Each process in a multiprocessing context gets its own
    singleton instance, making this safe for concurrent execution.

    Example:
    ```python
        class MyClass(Singleton):
            def __init__(self, value):
                self.value = value

        # Within the same process:
        obj1 = MyClass(10)
        obj2 = MyClass(20)
        assert obj1 is obj2  # Both refer to the same instance
        assert obj1.value == 10  # Value remains unchanged from first initialization

        # In a different process, a new singleton instance will be created
    ```
    """

    @classmethod
    def _reset_singleton(cls) -> None:
        """Reset the singleton instance (for testing only).

        Clears the singleton instance, allowing a fresh one to be created on
        the next instantiation. Also cleans up the associated lock.
        """
        key = (cls, os.getpid())
        SingletonMeta._instances.pop(key, None)
        SingletonMeta._instances_locks.pop(key, None)


def clear_all_singletons() -> None:
    """Clear all singleton instances cached by SingletonMeta for the current process.

    This is useful for test cleanup to ensure singleton instances
    don't leak between tests.
    """
    pid = os.getpid()
    keys_to_remove = [key for key in SingletonMeta._instances if key[1] == pid]
    for key in keys_to_remove:
        SingletonMeta._instances.pop(key, None)
        SingletonMeta._instances_locks.pop(key, None)
