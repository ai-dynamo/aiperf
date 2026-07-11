// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wall-vs-virtual clock abstraction.
//!
//! A single [`Clock`] trait is implemented by:
//!   - [`RealClock`] — real (monotonic) time; `sleep` actually waits.
//!   - [`SimClock`] — a virtual discrete-event clock advanced explicitly, so a
//!     run completes in simulated time with no wall-clock waits.
//!
//! The same async executor runs identically on either clock — the foundation
//! for driving one front-end both live (real time) and simulated (virtual time).

pub mod clock;
pub mod real_clock;
pub mod sim_clock;

pub use clock::Clock;
pub use real_clock::RealClock;
pub use sim_clock::SimClock;
