// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Item-and-byte resource accounting for native streaming stages.

use std::{
    fmt,
    io::{self, Write},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

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

/// Exact item-and-byte charge for one sub-reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BudgetCharge {
    /// Objects retained by the sub-reservation.
    pub items: usize,
    /// Bytes retained by the sub-reservation.
    pub bytes: usize,
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
    /// The requested capacity is currently in use.
    CapacityUnavailable,
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
    /// A leased write buffer was consumed without filling its exact charge.
    PartialLeasedBuffer {
        /// Byte capacity charged for the buffer.
        charged_bytes: usize,
        /// Bytes actually written into the buffer.
        written_bytes: usize,
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

    /// Acquire available item and byte capacity without waiting.
    ///
    /// Item capacity is attempted before byte capacity. If byte capacity is
    /// unavailable, the item permit is returned before this method reports the
    /// refusal.
    pub fn try_acquire(&self, items: usize, bytes: usize) -> Result<BudgetLease, BudgetError> {
        let item_permits = checked_permit_count(items)?;
        let byte_permits = checked_permit_count(bytes)?;
        if items > self.inner.limits.max_items || bytes > self.inner.limits.max_bytes {
            return Err(BudgetError::RequestExceedsCapacity);
        }

        let item_permit = Arc::clone(&self.inner.item_semaphore)
            .try_acquire_many_owned(item_permits)
            .map_err(map_try_acquire_error)?;
        let byte_permit = Arc::clone(&self.inner.byte_semaphore)
            .try_acquire_many_owned(byte_permits)
            .map_err(map_try_acquire_error)?;
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

    /// Atomically acquire two exact sub-reservations.
    ///
    /// The combined charge is acquired once and synchronously subdivided, so
    /// neither sub-reservation can be held while waiting for the other.
    pub async fn acquire_pair(
        &self,
        first: BudgetCharge,
        second: BudgetCharge,
    ) -> Result<(BudgetLease, BudgetLease), BudgetError> {
        let combined_items = first
            .items
            .checked_add(second.items)
            .ok_or(BudgetError::AccountingOverflow)?;
        let combined_bytes = first
            .bytes
            .checked_add(second.bytes)
            .ok_or(BudgetError::AccountingOverflow)?;
        let mut first_lease = self.acquire(combined_items, combined_bytes).await?;
        let second_lease = first_lease.split_off(second.items, second.bytes)?;
        Ok((first_lease, second_lease))
    }

    /// Close both capacity dimensions and wake pending acquisitions.
    pub fn close(&self) {
        self.inner.item_semaphore.close();
        self.inner.byte_semaphore.close();
    }

    /// Borrow the authored capacity this budget enforces.
    ///
    /// Observability reports a peak beside the limit it was bounded by, which is
    /// what makes a high-water mark evidence of boundedness rather than a number.
    #[must_use]
    pub fn limits(&self) -> BudgetLimits {
        self.inner.limits
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

    /// Move an exact sub-reservation into a separate lease.
    ///
    /// This divides the already-owned permits and recorded charge without
    /// releasing or reacquiring capacity.
    pub fn split_off(&mut self, items: usize, bytes: usize) -> Result<BudgetLease, BudgetError> {
        if items > self.charged_items || bytes > self.charged_bytes {
            return Err(BudgetError::CannotGrowLease);
        }

        let item_permit = self
            .item_permit
            .split(items)
            .ok_or(BudgetError::AccountingOverflow)?;
        let byte_permit = match self.byte_permit.split(bytes) {
            Some(permit) => permit,
            None => {
                self.item_permit.merge(item_permit);
                return Err(BudgetError::AccountingOverflow);
            }
        };

        self.charged_items -= items;
        self.charged_bytes -= bytes;
        Ok(BudgetLease {
            inner: Arc::clone(&self.inner),
            item_permit,
            byte_permit,
            charged_items: items,
            charged_bytes: bytes,
        })
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

/// Worst-case node-occupancy factor applied to an ordered-map entry's key and
/// value bytes.
///
/// `alloc::collections::BTreeMap` splits a full `CAPACITY == 11` node into
/// halves that retain `MIN_LEN_AFTER_SPLIT == 5` entries, so one live entry can
/// hold down `ceil(11 / 5) == 3` slots of node storage.
const ORDERED_MAP_NODE_OCCUPANCY_FACTOR: usize = 3;

/// Structural bytes one ordered-map entry retains beyond its key and value.
///
/// `BTreeMap` exposes no capacity API, so its node storage cannot be measured
/// the way a `Vec` or `String` capacity can. This is the authored per-entry
/// structural overhead used for charging, derived from the std B-tree node
/// shape as an upper bound accounting for worst-case BTreeMap node occupancy
/// after splits.
pub const ORDERED_MAP_ENTRY_OVERHEAD_BYTES: usize = 32;

/// Upper-bound structural charge for one ordered-map entry of `K` and `V`.
///
/// A `BTreeMap` node holds `CAPACITY == 11` key/value slots but retains as few
/// as `MIN_LEN_AFTER_SPLIT == 5` live entries immediately after a split, so a
/// single live entry can retain up to `ceil(11 / 5) == 3` slots' worth of node
/// storage. The key/value term is scaled by that factor so the charge bounds
/// worst-case node occupancy rather than assuming a densely packed node.
///
/// This covers the inline key, the inline value, and the authored per-entry
/// node overhead. Heap owned *behind* `K` or `V` is not included; add it with
/// [`checked_sum`].
///
/// # Errors
///
/// Returns [`BudgetError::AccountingOverflow`] when the sum is not representable.
pub fn ordered_map_entry_bytes<K, V>() -> Result<usize, BudgetError> {
    std::mem::size_of::<K>()
        .checked_add(std::mem::size_of::<V>())
        .and_then(|bytes| bytes.checked_mul(ORDERED_MAP_NODE_OCCUPANCY_FACTOR))
        .and_then(|bytes| bytes.checked_add(ORDERED_MAP_ENTRY_OVERHEAD_BYTES))
        .ok_or(BudgetError::AccountingOverflow)
}

/// Return the exact retained charge for one contiguous ring buffer of `T`.
///
/// # Errors
///
/// Returns [`BudgetError::AccountingOverflow`] when the product is not
/// representable.
pub fn ring_buffer_bytes<T>(capacity: usize) -> Result<usize, BudgetError> {
    capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(BudgetError::AccountingOverflow)
}

/// Sum exact charges with checked arithmetic.
///
/// # Errors
///
/// Returns [`BudgetError::AccountingOverflow`] on the first non-representable
/// partial sum.
pub fn checked_sum(parts: impl IntoIterator<Item = usize>) -> Result<usize, BudgetError> {
    parts
        .into_iter()
        .try_fold(0_usize, |total, part| total.checked_add(part))
        .ok_or(BudgetError::AccountingOverflow)
}

impl Drop for BudgetLease {
    fn drop(&mut self) {
        self.inner.release(self.charged_items, self.charged_bytes);
    }
}

/// Move-only write buffer whose allocated capacity is exactly its budget charge.
///
/// The buffer is allocated once, after admission, at the exact size its lease
/// paid for. [`Write`] refuses a byte past that capacity, so no producer can
/// outrun its admission and no reallocation can silently double the realized
/// footprint. The buffer and its lease are inseparable: the only way to obtain
/// the bytes is [`LeasedByteBuffer::into_full`], which returns the lease too.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::budget::LeasedByteBuffer;
/// # fn cannot_separate(value: LeasedByteBuffer) {
/// let _buffer = value.buffer;
/// let _lease = value.lease;
/// # }
/// ```
#[derive(Debug)]
pub struct LeasedByteBuffer {
    buffer: Vec<u8>,
    lease: BudgetLease,
}

impl LeasedByteBuffer {
    /// Allocate exactly the byte capacity the lease already paid for.
    ///
    /// # Errors
    ///
    /// Returns [`BudgetError::InvalidFragmentItemCharge`] unless the lease
    /// charges exactly one item, and [`BudgetError::AccountingOverflow`] when
    /// the exact reservation cannot be allocated. Both paths consume and drop
    /// the lease, so no charge outlives the refusal.
    pub fn with_exact_capacity(lease: BudgetLease) -> Result<Self, BudgetError> {
        if lease.charged_items() != 1 {
            return Err(BudgetError::InvalidFragmentItemCharge {
                charged_items: lease.charged_items(),
            });
        }
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(lease.charged_bytes())
            .map_err(|_| BudgetError::AccountingOverflow)?;
        Ok(Self { buffer, lease })
    }

    /// Return the byte capacity charged for this buffer.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }

    /// Return the bytes written so far.
    #[must_use]
    pub fn written_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Return the bytes still acceptable before the charge is exhausted.
    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        self.lease.charged_bytes() - self.buffer.len()
    }

    /// Consume an exactly filled buffer, returning compact bytes and the lease.
    ///
    /// Requiring the buffer to be exactly full is what makes the charge exact:
    /// `into_boxed_slice` on a `Vec` whose length equals its capacity never
    /// reallocates, so the returned allocation matches the charge.
    ///
    /// # Errors
    ///
    /// Returns [`BudgetError::PartialLeasedBuffer`] when fewer bytes were
    /// written than charged; the charge is released as the refusal returns.
    pub fn into_full(self) -> Result<(Box<[u8]>, BudgetLease), BudgetError> {
        if self.buffer.len() != self.lease.charged_bytes() {
            return Err(BudgetError::PartialLeasedBuffer {
                charged_bytes: self.lease.charged_bytes(),
                written_bytes: self.buffer.len(),
            });
        }
        Ok((self.buffer.into_boxed_slice(), self.lease))
    }
}

impl Write for LeasedByteBuffer {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if data.len() > self.remaining_capacity() {
            // Refusing here is the admission bound: a writer that would exceed
            // its charge fails instead of reallocating past it.
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "leased byte buffer capacity exhausted",
            ));
        }
        self.buffer.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn checked_permit_count(count: usize) -> Result<u32, BudgetError> {
    if count > Semaphore::MAX_PERMITS {
        return Err(BudgetError::PermitCountTooLarge);
    }
    u32::try_from(count).map_err(|_| BudgetError::PermitCountTooLarge)
}

fn map_try_acquire_error(error: TryAcquireError) -> BudgetError {
    match error {
        TryAcquireError::Closed => BudgetError::Closed,
        TryAcquireError::NoPermits => BudgetError::CapacityUnavailable,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_map_entry_bytes_is_checked() {
        let bytes = ordered_map_entry_bytes::<u64, u64>()
            .unwrap_or_else(|error| panic!("representable entry charge: {error}"));
        assert_eq!(
            bytes,
            ORDERED_MAP_NODE_OCCUPANCY_FACTOR * 16 + ORDERED_MAP_ENTRY_OVERHEAD_BYTES
        );
    }

    #[test]
    fn ring_buffer_bytes_is_checked() {
        let bytes = ring_buffer_bytes::<u32>(8)
            .unwrap_or_else(|error| panic!("representable ring charge: {error}"));
        assert_eq!(bytes, 32);
        assert_eq!(
            ring_buffer_bytes::<u64>(usize::MAX),
            Err(BudgetError::AccountingOverflow)
        );
    }

    #[test]
    fn checked_sum_reports_accounting_overflow() {
        assert_eq!(checked_sum([1_usize, 2, 3]), Ok(6));
        assert_eq!(
            checked_sum([usize::MAX, 1]),
            Err(BudgetError::AccountingOverflow)
        );
    }

    #[test]
    fn leased_buffer_capacity_equals_charge() {
        let budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: 4,
            max_bytes: 64,
        })
        .unwrap_or_else(|error| panic!("valid budget: {error}"));
        let lease = budget
            .try_acquire(1, 8)
            .unwrap_or_else(|error| panic!("available lease: {error}"));
        let mut buffer = LeasedByteBuffer::with_exact_capacity(lease)
            .unwrap_or_else(|error| panic!("exact buffer: {error}"));

        assert_eq!(buffer.charged_bytes(), 8);
        assert_eq!(buffer.remaining_capacity(), 8);
        buffer
            .write_all(b"abcdefgh")
            .unwrap_or_else(|error| panic!("exact fill: {error}"));
        assert_eq!(buffer.written_bytes(), 8);

        let (bytes, lease) = buffer
            .into_full()
            .unwrap_or_else(|error| panic!("exactly filled: {error}"));
        assert_eq!(bytes.as_ref(), b"abcdefgh");
        assert_eq!(bytes.len(), lease.charged_bytes());
        assert_eq!(budget.snapshot().used_bytes, 8);
        drop(lease);
        assert_eq!(budget.snapshot().used_bytes, 0);
    }

    #[test]
    fn leased_buffer_write_past_charge_fails() {
        let budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: 4,
            max_bytes: 64,
        })
        .unwrap_or_else(|error| panic!("valid budget: {error}"));
        let mut buffer = LeasedByteBuffer::with_exact_capacity(
            budget
                .try_acquire(1, 8)
                .unwrap_or_else(|error| panic!("available lease: {error}")),
        )
        .unwrap_or_else(|error| panic!("exact buffer: {error}"));

        buffer
            .write_all(b"abcd")
            .unwrap_or_else(|error| panic!("partial write fits: {error}"));
        assert!(buffer.write_all(b"toolong").is_err());
        assert_eq!(buffer.written_bytes(), 4);
        assert_eq!(budget.snapshot().used_bytes, 8);
    }

    #[test]
    fn into_full_rejects_partial_buffer() {
        let budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: 4,
            max_bytes: 64,
        })
        .unwrap_or_else(|error| panic!("valid budget: {error}"));
        let mut buffer = LeasedByteBuffer::with_exact_capacity(
            budget
                .try_acquire(1, 8)
                .unwrap_or_else(|error| panic!("available lease: {error}")),
        )
        .unwrap_or_else(|error| panic!("exact buffer: {error}"));
        buffer
            .write_all(b"abc")
            .unwrap_or_else(|error| panic!("partial write: {error}"));

        assert_eq!(
            buffer.into_full().err(),
            Some(BudgetError::PartialLeasedBuffer {
                charged_bytes: 8,
                written_bytes: 3,
            })
        );
        assert_eq!(budget.snapshot().used_items, 0);
        assert_eq!(budget.snapshot().used_bytes, 0);
    }
}
