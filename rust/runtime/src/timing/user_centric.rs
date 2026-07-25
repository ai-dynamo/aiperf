// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! User-centric rate schedule — the pure virtual-history math.
//!
//! At `t=0`, deterministic virtual history places users at different session
//! ages so cache pressure begins at steady state. This module is RNG- and
//! clock-free.
//!
//! Each of the `num_users` users is assigned a virtual "age" spreading them
//! across the *session lifetime* (turns from first to last, i.e.
//! `session_turns - 1`): user 1 (oldest) is virtually done and replaced
//! immediately by a fresh user firing at `t=0`; user N (youngest) has the most
//! turns remaining. Firing order is staggered so users start and finish
//! throughout the run.
//!
//! Timing knobs (they set the cache pressure — the exact formulas matter):
//! - `stagger = 1 / rate` — smallest gap between two users' first turns.
//! - `turn_gap = num_users / rate` — per-user inter-turn gap
//!   (`qps = num_users / turn_gap`).
//!
//! All times use integer nanoseconds.

use crate::timing::secs_to_ns;

/// Greatest common divisor via the Euclidean algorithm.
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Smallest integer `step` in `2..n` that is coprime with `n`, else `1`.
///
/// Iterating positions `(i * step) % n` for `i in 0..n` visits every slot
/// exactly once iff `step` is coprime with `n`. Returns `1` for `n <= 2`, where
/// no such `step > 1` exists (the range `2..n` is empty).
pub fn find_alternate_spacing_step(n: usize) -> usize {
    for step in 2..n {
        if gcd(n, step) == 1 {
            return step;
        }
    }
    1
}

/// One pre-seeded initial user in the steady-state schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InitialUser {
    /// Sequential id assigned in order (ids of "virtually done" users are
    /// burned, so emitted ids may skip values).
    pub user_id: u64,
    /// Position in the initial stagger sequence (`0` fires first).
    pub order: usize,
    /// Turns this user still has to send (reduced by its virtual history).
    pub max_turns: usize,
    /// Offset of this user's first send from phase start = `order * stagger`.
    pub first_send_offset_ns: i64,
}

/// Deterministic steady-state seeding plan for the user-centric strategy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UserCentricPlan {
    /// Smallest gap between two users' first turns (`1 / rate`), in ns.
    pub stagger_ns: i64,
    /// Per-user inter-turn gap (`num_users / rate`), in ns.
    pub turn_gap_ns: i64,
    /// Initial users in emission (id) order; the last entry is always the fresh
    /// `order = 0` replacement for the user that finished before `t=0`.
    pub initial_users: Vec<InitialUser>,
}

/// Compute the user-centric steady-state seeding plan.
///
/// `num_users` and `request_rate` must be positive; `avg_session_turns` is the
/// rounded average turn count of the dataset. Pure and deterministic — no RNG,
/// no clock.
///
/// # Panics
/// Panics if `num_users == 0` or `request_rate` is not `> 0` and finite.
pub fn plan_user_centric(
    num_users: usize,
    avg_session_turns: usize,
    request_rate: f64,
) -> UserCentricPlan {
    assert!(
        num_users > 0,
        "num_users must be > 0 for user-centric rate mode"
    );
    assert!(
        request_rate.is_finite() && request_rate > 0.0,
        "request_rate must be positive and finite for user-centric rate mode, got {request_rate}"
    );

    let stagger_secs = 1.0 / request_rate;
    let turn_gap_secs = num_users as f64 / request_rate;
    let stagger_ns = secs_to_ns(stagger_secs);
    let turn_gap_ns = secs_to_ns(turn_gap_secs);

    // Session lifetime = gaps between turns (session_turns - 1), floored at 1 so
    // even single-turn sessions get spacing. saturating_sub guards avg == 0.
    let session_lifetime = std::cmp::max(1, avg_session_turns.saturating_sub(1));

    // When num_users and session_lifetime share a factor, `virtual_age % n`
    // collides; fall back to a coprime step so every slot is unique.
    let use_alternate_spacing = gcd(num_users, session_lifetime) > 1;
    let spacing_step = if use_alternate_spacing {
        find_alternate_spacing_step(num_users)
    } else {
        1
    };

    let mut initial_users: Vec<InitialUser> = Vec::with_capacity(num_users);
    let mut next_user_id: u64 = 1;

    for i in 0..num_users {
        // Older users (small i) have a high virtual age -> most turns already
        // completed before t=0. This mixes almost-done and just-started users.
        let virtual_age = (num_users - i) * session_lifetime;
        let session_age = virtual_age / num_users;
        // session_age <= session_lifetime, with equality only at i == 0.
        let turns_to_send = session_lifetime as i64 - session_age as i64;

        if turns_to_send <= 0 {
            // Burn virtually completed IDs to preserve subsequent ordering.
            next_user_id += 1;
            continue;
        }

        let slot_index = if use_alternate_spacing {
            (i * spacing_step) % num_users
        } else {
            virtual_age % num_users
        };
        let starting_order = num_users - slot_index;

        let user_id = next_user_id;
        next_user_id += 1;
        initial_users.push(InitialUser {
            user_id,
            order: starting_order,
            max_turns: turns_to_send as usize,
            first_send_offset_ns: secs_to_ns(starting_order as f64 * stagger_secs),
        });
    }

    // Always add a fresh user at order 0 (first send at t=0) with all turns,
    // replacing the user that was virtually done before the run started.
    initial_users.push(InitialUser {
        user_id: next_user_id,
        order: 0,
        max_turns: avg_session_turns,
        first_send_offset_ns: 0,
    });

    UserCentricPlan {
        stagger_ns,
        turn_gap_ns,
        initial_users,
    }
}

/// Absolute spawn time of the replacement for a user spawned at `prev_spawn_ns`.
///
/// Open-loop schedule: `prev_spawn_ns + max_turns * turn_gap_ns`, independent of
/// response times.
pub fn next_replacement_spawn_ns(prev_spawn_ns: i64, max_turns: usize, turn_gap_ns: i64) -> i64 {
    prev_spawn_ns + max_turns as i64 * turn_gap_ns
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn docstring_example_15_users_20_turns_1qps() {
        let plan = plan_user_centric(15, 20, 1.0);

        // turn_gap = num_users / rate = 15s; stagger = 1 / rate = 1s.
        assert_eq!(plan.turn_gap_ns, 15_000_000_000);
        assert_eq!(plan.stagger_ns, 1_000_000_000);

        // 14 seeded users (user 1 is virtually done) + 1 fresh replacement.
        assert_eq!(plan.initial_users.len(), 15);

        // Fresh replacement is last: order 0, full turns, id 16.
        let fresh = *plan.initial_users.last().unwrap();
        assert_eq!(fresh.order, 0);
        assert_eq!(fresh.max_turns, 20);
        assert_eq!(fresh.user_id, 16);
        assert_eq!(fresh.first_send_offset_ns, 0);

        // Orders are a permutation of 0..15 (every stagger slot filled once).
        let orders: HashSet<usize> = plan.initial_users.iter().map(|u| u.order).collect();
        assert_eq!(orders.len(), 15);
        assert_eq!(orders, (0..15).collect::<HashSet<usize>>());

        // First send offset always equals order * stagger.
        for u in &plan.initial_users {
            assert_eq!(u.first_send_offset_ns, u.order as i64 * plan.stagger_ns);
        }

        // Spot-check a couple of rows from the docstring table.
        // User id 2 -> 2 turns, fires at 4s (order 4).
        let u2 = plan.initial_users.iter().find(|u| u.user_id == 2).unwrap();
        assert_eq!(u2.max_turns, 2);
        assert_eq!(u2.order, 4);
        // User id 3 -> 3 turns, fires at 8s (order 8).
        let u3 = plan.initial_users.iter().find(|u| u.user_id == 3).unwrap();
        assert_eq!(u3.max_turns, 3);
        assert_eq!(u3.order, 8);
    }

    #[test]
    fn turn_gap_and_stagger_formulas() {
        // turn_gap = num_users / rate, stagger = 1 / rate.
        let plan = plan_user_centric(10, 5, 2.0);
        assert_eq!(plan.stagger_ns, secs_to_ns(1.0 / 2.0)); // 0.5s
        assert_eq!(plan.turn_gap_ns, secs_to_ns(10.0 / 2.0)); // 5s
        assert_eq!(plan.stagger_ns, 500_000_000);
        assert_eq!(plan.turn_gap_ns, 5_000_000_000);
    }

    #[test]
    fn alternate_spacing_step_is_smallest_coprime() {
        assert_eq!(find_alternate_spacing_step(1), 1);
        assert_eq!(find_alternate_spacing_step(2), 1);
        assert_eq!(find_alternate_spacing_step(15), 2);
        assert_eq!(find_alternate_spacing_step(6), 5);
        assert_eq!(find_alternate_spacing_step(9), 2);
    }

    #[test]
    fn alternate_spacing_path_yields_unique_orders() {
        // session_lifetime = avg_session_turns - 1 = 15 shares gcd 15 with
        // num_users 15, so the alternate (coprime-step) path is taken.
        let plan = plan_user_centric(15, 16, 1.0);
        assert_eq!(find_alternate_spacing_step(15), 2);

        // Orders must still be a unique permutation covering every slot.
        let orders: HashSet<usize> = plan.initial_users.iter().map(|u| u.order).collect();
        assert_eq!(orders.len(), plan.initial_users.len());
        assert_eq!(orders, (0..15).collect::<HashSet<usize>>());
    }

    #[test]
    fn id_accounting_burns_done_users_but_keeps_order() {
        // Exactly one user (i=0) is virtually done: its id (1) is burned, so the
        // first emitted user starts at id 2 and ids stay strictly increasing.
        let plan = plan_user_centric(15, 20, 1.0);

        assert_eq!(plan.initial_users[0].user_id, 2);
        // Fresh replacement takes num_users + 1 (all prior ids consumed once).
        assert_eq!(plan.initial_users.last().unwrap().user_id, 16);

        let ids: Vec<u64> = plan.initial_users.iter().map(|u| u.user_id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "ids must be emitted in increasing order");
        // Seeded users are the contiguous run 2..=15, then the fresh 16.
        assert_eq!(ids, (2..=16).collect::<Vec<u64>>());
    }

    #[test]
    fn single_user_still_seeds_a_fresh_replacement() {
        // num_users = 1: the lone user is virtually done, leaving only the fresh
        // t=0 replacement.
        let plan = plan_user_centric(1, 3, 1.0);
        assert_eq!(plan.initial_users.len(), 1);
        let only = plan.initial_users[0];
        assert_eq!(only.order, 0);
        assert_eq!(only.max_turns, 3);
        // id 1 burned by the done user -> fresh user is id 2.
        assert_eq!(only.user_id, 2);
    }

    #[test]
    fn replacement_spawn_is_open_loop() {
        // next = prev + max_turns * turn_gap.
        assert_eq!(
            next_replacement_spawn_ns(1_000, 4, 15_000_000_000),
            1_000 + 4 * 15_000_000_000
        );
        assert_eq!(next_replacement_spawn_ns(0, 0, 5_000), 0);
    }

    #[test]
    #[should_panic(expected = "num_users must be > 0")]
    fn zero_users_panics() {
        plan_user_centric(0, 5, 1.0);
    }

    #[test]
    #[should_panic(expected = "request_rate must be positive")]
    fn non_positive_rate_panics() {
        plan_user_centric(5, 5, 0.0);
    }
}
