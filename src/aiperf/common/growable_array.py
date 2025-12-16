# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-allocated numpy arrays with automatic growth."""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

__all__ = ["GrowableArray", "Growable2DArray"]


class GrowableArray:
    """Pre-allocated 1D numpy array with automatic capacity doubling.

    Provides amortized O(1) append. Memory overhead bounded to 2x minimum.

    Example:
        >>> arr = GrowableArray(initial_capacity=4, dtype=np.int64)
        >>> for i in range(10):
        ...     arr.append(i)
        >>> arr.data
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> len(arr), arr.capacity
        (10, 16)
    """

    __slots__ = ("_data", "_size", "_sum")

    def __init__(
        self,
        initial_capacity: int = 256,
        dtype: DTypeLike = np.float64,
        track_sum: bool = False,
    ) -> None:
        """Initialize with pre-allocated capacity.

        Args:
            initial_capacity: Initial array capacity. Must be positive.
            dtype: NumPy dtype for the array.
            track_sum: If True, maintain a running sum for O(1) mean calculation.
        """
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")
        self._data: NDArray = np.empty(initial_capacity, dtype=dtype)
        self._size: int = 0
        self._sum: float | None = 0.0 if track_sum else None

    def append(self, value: float | int) -> None:
        """Append a single value, growing capacity if necessary.

        Automatically doubles capacity when full, providing amortized O(1)
        append performance. Updates running sum if track_sum enabled.

        Args:
            value: Numeric value to append (must be compatible with array dtype)
        """
        if self._size >= len(self._data):
            self._grow()
        self._data[self._size] = value
        self._size += 1
        if self._sum is not None:
            self._sum += value

    def extend(self, values: NDArray) -> None:
        """Append multiple values from an array.

        More efficient than repeated append() calls for bulk insertion.
        Grows capacity in a single reallocation to fit all values if needed.
        Updates running sum if track_sum enabled.

        Args:
            values: NumPy array of values to append (must be compatible with array dtype)
        """
        n = len(values)
        if n == 0:
            return
        self._ensure_capacity(n)
        self._data[self._size : self._size + n] = values
        self._size += n
        if self._sum is not None:
            self._sum += float(values.sum())

    def _ensure_capacity(self, additional: int) -> None:
        """Ensure capacity for `additional` more elements.

        Grows capacity by doubling repeatedly until sufficient space available.
        More efficient than doubling incrementally when bulk inserting.

        Args:
            additional: Number of additional elements needed
        """
        required = self._size + additional
        if required > len(self._data):
            new_cap = len(self._data)
            while new_cap < required:
                new_cap *= 2
            self._grow_to(new_cap)

    def _grow(self) -> None:
        """Double the capacity.

        Standard doubling strategy provides amortized O(1) append while
        keeping memory overhead bounded to 2x minimum required.
        """
        self._grow_to(len(self._data) * 2)

    def _grow_to(self, new_capacity: int) -> None:
        """Grow to a specific capacity.

        Allocates new array and copies existing data. Used by both _grow()
        and _ensure_capacity() to avoid code duplication.

        Args:
            new_capacity: New array capacity (must be >= current size)
        """
        new_data = np.empty(new_capacity, dtype=self._data.dtype)
        new_data[: self._size] = self._data[: self._size]
        self._data = new_data

    @property
    def data(self) -> NDArray:
        """View of stored data (no copy).

        Returns a view of the active portion of the underlying array.
        Does not copy data, so modifications to returned array affect storage.

        Returns:
            NumPy array view of shape (size,) containing stored elements
        """
        return self._data[: self._size]

    @property
    def capacity(self) -> int:
        """Current allocated capacity.

        May be larger than size due to pre-allocation. Maximum overhead is 2x.

        Returns:
            Total allocated capacity in elements
        """
        return len(self._data)

    @property
    def sum(self) -> float | None:
        """Running sum if tracking enabled, else None.

        O(1) access when track_sum=True, avoiding repeated summation.

        Returns:
            Cumulative sum of all appended values, or None if tracking disabled
        """
        return self._sum

    @property
    def mean(self) -> float | None:
        """Running mean if tracking enabled and non-empty, else None.

        O(1) computation when track_sum=True.

        Returns:
            Mean of all stored values (sum/size), or None if tracking disabled or empty
        """
        if self._sum is None or self._size == 0:
            return None
        return self._sum / self._size

    def __len__(self) -> int:
        """Number of stored elements.

        Returns:
            Count of elements currently stored (not capacity)
        """
        return self._size

    def clear(self) -> None:
        """Reset to empty (keeps allocated capacity).

        Resets size and sum to zero without deallocating memory.
        Subsequent appends reuse the existing allocation.
        """
        self._size = 0
        if self._sum is not None:
            self._sum = 0.0

    @overload
    def __getitem__(self, index: int) -> float | int: ...

    @overload
    def __getitem__(self, index: slice) -> NDArray: ...

    def __getitem__(self, index: int | slice) -> float | int | NDArray:
        """Get element(s) by index or slice.

        Supports negative indexing and slice notation.

        Args:
            index: Integer index or slice object

        Returns:
            Single value for int index, or view array for slice

        Raises:
            IndexError: If index out of bounds
        """
        return self.data[index]

    def __setitem__(self, index: int, value: float | int) -> None:
        """Set element by index. Updates running sum if tracking enabled.

        Supports negative indexing. Running sum is automatically adjusted
        to account for the value change when track_sum=True.

        Args:
            index: Integer index (negative indexes supported)
            value: New value to set

        Raises:
            IndexError: If index out of bounds
        """
        if index < 0:
            index = self._size + index
        if index < 0 or index >= self._size:
            raise IndexError("index out of range")
        if self._sum is not None:
            self._sum -= self._data[index]
            self._sum += value
        self._data[index] = value


class Growable2DArray:
    """Pre-allocated 2D numpy array that grows along axis 0.

    Column count is fixed at construction. Row capacity doubles when full.

    Example:
        >>> arr = Growable2DArray(n_columns=3, initial_capacity=4)
        >>> arr.append(np.array([1.0, 2.0, 3.0]))
        >>> arr.append(np.array([4.0, 5.0, 6.0]))
        >>> arr.data
        array([[1., 2., 3.],
               [4., 5., 6.]])
    """

    __slots__ = ("_data", "_size")

    def __init__(
        self,
        n_columns: int,
        initial_capacity: int = 256,
        dtype: DTypeLike = np.float64,
    ) -> None:
        """Initialize with pre-allocated capacity.

        Args:
            n_columns: Fixed number of columns.
            initial_capacity: Initial row capacity. Must be positive.
            dtype: NumPy dtype for the array.
        """
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")
        if n_columns <= 0:
            raise ValueError("n_columns must be positive")
        self._data: NDArray = np.empty((initial_capacity, n_columns), dtype=dtype)
        self._size: int = 0

    def append(self, row: NDArray) -> None:
        """Append a single row, growing capacity if necessary.

        Automatically doubles row capacity when full, providing amortized O(1)
        append performance. Row must match the fixed column count.

        Args:
            row: 1D NumPy array of shape (n_columns,) to append

        Raises:
            ValueError: If row shape doesn't match n_columns (handled by NumPy)
        """
        if self._size >= len(self._data):
            self._grow()
        self._data[self._size] = row
        self._size += 1

    def _grow(self) -> None:
        """Double the row capacity.

        Allocates new 2D array with doubled row capacity and copies existing
        data. Column count remains fixed. Provides amortized O(1) row append.
        """
        new_cap = len(self._data) * 2
        new_data = np.empty((new_cap, self.n_columns), dtype=self._data.dtype)
        new_data[: self._size] = self._data[: self._size]
        self._data = new_data

    @property
    def data(self) -> NDArray:
        """View of stored data (no copy).

        Returns a view of the active portion of the underlying 2D array.
        Does not copy data, so modifications to returned array affect storage.

        Returns:
            NumPy array view of shape (size, n_columns) containing stored rows
        """
        return self._data[: self._size]

    @property
    def capacity(self) -> int:
        """Current allocated row capacity.

        May be larger than size due to pre-allocation. Maximum overhead is 2x.

        Returns:
            Total allocated row capacity
        """
        return len(self._data)

    @property
    def n_columns(self) -> int:
        """Number of columns (fixed at construction).

        Returns:
            Column count (second dimension of array)
        """
        return self._data.shape[1]

    def __len__(self) -> int:
        """Number of stored rows.

        Returns:
            Count of rows currently stored (not capacity)
        """
        return self._size

    def clear(self) -> None:
        """Reset to empty (keeps allocated capacity).

        Resets row count to zero without deallocating memory.
        Subsequent appends reuse the existing allocation.
        """
        self._size = 0

    @overload
    def __getitem__(self, index: int) -> NDArray: ...

    @overload
    def __getitem__(self, index: slice) -> NDArray: ...

    def __getitem__(self, index: int | slice) -> NDArray:
        """Get row(s) by index or slice.

        Supports negative indexing and slice notation.

        Args:
            index: Integer index or slice object

        Returns:
            Single row (1D array) for int index, or 2D array view for slice

        Raises:
            IndexError: If index out of bounds
        """
        return self.data[index]

    def __setitem__(self, index: int, value: NDArray) -> None:
        """Set row by index.

        Supports negative indexing. Row must match the fixed column count.

        Args:
            index: Integer index (negative indexes supported)
            value: 1D array of shape (n_columns,) to set as new row

        Raises:
            IndexError: If index out of bounds
            ValueError: If value shape doesn't match n_columns (handled by NumPy)
        """
        if index < 0:
            index = self._size + index
        if index < 0 or index >= self._size:
            raise IndexError("index out of range")
        self._data[index] = value
