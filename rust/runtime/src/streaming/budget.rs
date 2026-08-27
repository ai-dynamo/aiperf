// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Item-and-byte resource accounting for native streaming stages.

use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const ITEM_SHIFT: u32 = u32::BITS;
const COUNT_MASK: u64 = u32::MAX as u64;

/// Fixed item and byte capacity for one streaming resource category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BudgetLimits {
    /// Maximum simultaneously retained objects.
    pub max_items: usize,
    /// Maximum simultaneously retained bytes.
    pub max_bytes: usize,
}

/// Current and peak resource use for one budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BudgetSnapshot {
    /// Currently charged objects.
    pub used_items: usize,
    /// Currently charged bytes.
    pub used_bytes: usize,
    /// Greatest observed item charge.
    pub high_water_items: usize,
    /// Greatest observed byte charge.
    pub high_water_bytes: usize,
}

/// Resource-budget validation or acquisition failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BudgetError {
    /// At least one configured capacity is zero.
    ZeroCapacity,
    /// A configured or requested permit count cannot be represented by Tokio.
    PermitCountTooLarge,
    /// A request exceeds the configured budget.
    RequestExceedsCapacity,
    /// The budget was closed.
    Closed,
    /// A lease can only shrink its charge.
    CannotGrowLease,
    /// A session fragment must retain exactly one item permit.
    InvalidFragmentItemCharge {
        /// Actual item charge on the consumed generic lease.
        charged_items: usize,
    },
    /// An action payload is larger than its distinct retained-content charges.
    ActionPayloadUndercharged {
        /// Bytes required by payload and spilled action-owned metadata.
        required_bytes: usize,
        /// Bytes covered by distinct retained-content leases.
        retained_bytes: usize,
    },
    /// Internal accounting could not represent a state transition.
    AccountingOverflow,
}

impl fmt::Display for BudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "streaming resource budget error: {self:?}")
    }
}

impl std::error::Error for BudgetError {}

#[derive(Debug)]
struct BudgetInner {
    limits: BudgetLimits,
    item_semaphore: Arc<Semaphore>,
    byte_semaphore: Arc<Semaphore>,
    current: AtomicU64,
    high_water_items: AtomicU64,
    high_water_bytes: AtomicU64,
}

impl BudgetInner {
    fn charge(&self, items: usize, bytes: usize) -> Result<(), BudgetError> {
        let mut observed = self.current.load(Ordering::Acquire);
        loop {
            let (used_items, used_bytes) = unpack_counts(observed);
            let next_items = used_items
                .checked_add(items)
                .ok_or(BudgetError::AccountingOverflow)?;
            let next_bytes = used_bytes
                .checked_add(bytes)
                .ok_or(BudgetError::AccountingOverflow)?;
            if next_items > self.limits.max_items || next_bytes > self.limits.max_bytes {
                return Err(BudgetError::AccountingOverflow);
            }
            let next = pack_counts(next_items, next_bytes)?;
            match self.current.compare_exchange_weak(
                observed,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.high_water_items
                        .fetch_max(next_items as u64, Ordering::Relaxed);
                    self.high_water_bytes
                        .fetch_max(next_bytes as u64, Ordering::Relaxed);
                    return Ok(());
                }
                Err(actual) => observed = actual,
            }
        }
    }

    fn release(&self, items: usize, bytes: usize) {
        let mut observed = self.current.load(Ordering::Acquire);
        loop {
            let (used_items, used_bytes) = unpack_counts(observed);
            let (Some(next_items), Some(next_bytes)) =
                (used_items.checked_sub(items), used_bytes.checked_sub(bytes))
            else {
                tracing::error!(
                    used_items,
                    used_bytes,
                    items,
                    bytes,
                    "budget accounting underflow"
                );
                return;
            };
            let Ok(next) = pack_counts(next_items, next_bytes) else {
                tracing::error!(
                    next_items,
                    next_bytes,
                    "budget counters are not representable"
                );
                return;
            };
            match self.current.compare_exchange_weak(
                observed,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(actual) => observed = actual,
            }
        }
    }
}

/// Shared item-and-byte resource budget.
#[derive(Clone, Debug)]
pub struct StreamingResourceBudget {
    inner: Arc<BudgetInner>,
}

impl StreamingResourceBudget {
    /// Construct a validated resource budget.
    pub fn new(limits: BudgetLimits) -> Result<Self, BudgetError> {
        if limits.max_items == 0 || limits.max_bytes == 0 {
            return Err(BudgetError::ZeroCapacity);
        }
        checked_permit_count(limits.max_items)?;
        checked_permit_count(limits.max_bytes)?;
        Ok(Self {
            inner: Arc::new(BudgetInner {
                limits,
                item_semaphore: Arc::new(Semaphore::new(limits.max_items)),
                byte_semaphore: Arc::new(Semaphore::new(limits.max_bytes)),
                current: AtomicU64::new(0),
                high_water_items: AtomicU64::new(0),
                high_water_bytes: AtomicU64::new(0),
            }),
        })
    }

    /// Wait for both item and byte capacity.
    ///
    /// Item capacity is always acquired before byte capacity. Cancellation or
    /// byte-acquisition failure drops the owned item permit before returning.
    pub async fn acquire(&self, items: usize, bytes: usize) -> Result<BudgetLease, BudgetError> {
        let item_permits = checked_permit_count(items)?;
        let byte_permits = checked_permit_count(bytes)?;
        if items > self.inner.limits.max_items || bytes > self.inner.limits.max_bytes {
            return Err(BudgetError::RequestExceedsCapacity);
        }

        let item_permit = Arc::clone(&self.inner.item_semaphore)
            .acquire_many_owned(item_permits)
            .await
            .map_err(|_| BudgetError::Closed)?;
        let byte_permit = Arc::clone(&self.inner.byte_semaphore)
            .acquire_many_owned(byte_permits)
            .await
            .map_err(|_| BudgetError::Closed)?;
        if self.inner.item_semaphore.is_closed() || self.inner.byte_semaphore.is_closed() {
            return Err(BudgetError::Closed);
        }

        self.inner.charge(items, bytes)?;
        Ok(BudgetLease {
            inner: Arc::clone(&self.inner),
            item_permit,
            byte_permit,
            charged_items: items,
            charged_bytes: bytes,
        })
    }

    /// Close both capacity dimensions and wake pending acquisitions.
    pub fn close(&self) {
        self.inner.item_semaphore.close();
        self.inner.byte_semaphore.close();
    }

    /// Read an atomic pair of current counters and independent peak counters.
    #[must_use]
    pub fn snapshot(&self) -> BudgetSnapshot {
        let (used_items, used_bytes) = unpack_counts(self.inner.current.load(Ordering::Acquire));
        let high_water_items = self.inner.high_water_items.load(Ordering::Relaxed) as usize;
        let high_water_bytes = self.inner.high_water_bytes.load(Ordering::Relaxed) as usize;
        BudgetSnapshot {
            used_items,
            used_bytes,
            high_water_items: high_water_items.max(used_items),
            high_water_bytes: high_water_bytes.max(used_bytes),
        }
    }
}

/// Move-only ownership of one item-and-byte budget charge.
#[derive(Debug)]
pub struct BudgetLease {
    inner: Arc<BudgetInner>,
    item_permit: OwnedSemaphorePermit,
    byte_permit: OwnedSemaphorePermit,
    charged_items: usize,
    charged_bytes: usize,
}

impl BudgetLease {
    /// Return the current item charge.
    #[must_use]
    pub fn charged_items(&self) -> usize {
        self.charged_items
    }

    /// Return the current byte charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.charged_bytes
    }

    /// Return excess capacity while retaining the requested smaller charge.
    pub fn shrink_to(&mut self, items: usize, bytes: usize) -> Result<(), BudgetError> {
        if items > self.charged_items || bytes > self.charged_bytes {
            return Err(BudgetError::CannotGrowLease);
        }

        let returned_items = self.charged_items - items;
        let returned_bytes = self.charged_bytes - bytes;
        let item_permit = self
            .item_permit
            .split(returned_items)
            .ok_or(BudgetError::AccountingOverflow)?;
        let byte_permit = match self.byte_permit.split(returned_bytes) {
            Some(permit) => permit,
            None => {
                self.item_permit.merge(item_permit);
                return Err(BudgetError::AccountingOverflow);
            }
        };

        self.charged_items = items;
        self.charged_bytes = bytes;
        self.inner.release(returned_items, returned_bytes);
        drop(item_permit);
        drop(byte_permit);
        Ok(())
    }
}

impl Drop for BudgetLease {
    fn drop(&mut self) {
        self.inner.release(self.charged_items, self.charged_bytes);
    }
}

fn checked_permit_count(count: usize) -> Result<u32, BudgetError> {
    if count > Semaphore::MAX_PERMITS {
        return Err(BudgetError::PermitCountTooLarge);
    }
    u32::try_from(count).map_err(|_| BudgetError::PermitCountTooLarge)
}

fn pack_counts(items: usize, bytes: usize) -> Result<u64, BudgetError> {
    let items = u32::try_from(items).map_err(|_| BudgetError::AccountingOverflow)?;
    let bytes = u32::try_from(bytes).map_err(|_| BudgetError::AccountingOverflow)?;
    Ok((u64::from(items) << ITEM_SHIFT) | u64::from(bytes))
}

fn unpack_counts(counts: u64) -> (usize, usize) {
    (
        (counts >> ITEM_SHIFT) as usize,
        (counts & COUNT_MASK) as usize,
    )
}
