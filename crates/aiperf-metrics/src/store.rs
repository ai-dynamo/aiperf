// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sparse numeric columns used by accumulator implementations.

/// A numeric column that keeps absence explicit instead of exporting NaN.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NumericColumn {
    values: Vec<Option<f64>>,
}

impl NumericColumn {
    /// Builds an empty numeric column.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a present finite value or absence for non-finite input.
    pub fn push_f64(&mut self, value: f64) {
        self.values.push(value.is_finite().then_some(value));
    }

    /// Appends an explicit missing value.
    pub fn push_absent(&mut self) {
        self.values.push(None);
    }

    /// Returns the value at `idx`.
    pub fn get(&self, idx: usize) -> Option<f64> {
        self.values.get(idx).and_then(|value| *value)
    }

    /// Number of rows in the column.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true when the column is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Small column store facade reserved for the full metrics engine.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColumnStore {
    /// Named numeric columns.
    pub numeric: Vec<(String, NumericColumn)>,
}

#[cfg(test)]
mod tests {
    use super::NumericColumn;

    #[test]
    fn numeric_column_scrubs_non_finite_values() {
        let mut column = NumericColumn::new();
        column.push_f64(1.0);
        column.push_f64(f64::NAN);
        assert_eq!(column.get(0), Some(1.0));
        assert_eq!(column.get(1), None);
    }
}
