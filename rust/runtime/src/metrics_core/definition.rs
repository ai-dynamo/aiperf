// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit newtypes for native and display metric values.

/// A value in a metric's native (math/SLA) unit.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Native(pub f64);

/// A value in a metric's human display unit.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Display(pub f64);

impl Native {
    pub const fn new(v: f64) -> Self {
        Self(v)
    }
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl Display {
    pub const fn new(v: f64) -> Self {
        Self(v)
    }
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl std::ops::Add for Native {
    type Output = Native;
    fn add(self, o: Native) -> Native {
        Native(self.0 + o.0)
    }
}

impl std::ops::Sub for Native {
    type Output = Native;
    fn sub(self, o: Native) -> Native {
        Native(self.0 - o.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_orders_and_adds_within_scale() {
        assert!(Native::new(2.0) > Native::new(1.0));
        assert_eq!((Native::new(1.0) + Native::new(2.0)).get(), 3.0);
    }
}
