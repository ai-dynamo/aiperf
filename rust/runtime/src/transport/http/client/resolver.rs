// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-injected DNS resolution and caching.
//!
//! This preserves the aiohttp event contract: a cache hit skips
//! lookup timing, while a miss records both the cache event and the resolver
//! bracket.

use std::cell::RefCell;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::rc::Rc;

use crate::clock::Clock;
use async_trait::async_trait;

use crate::transport::core::{ErrorDetails, ErrorKind, TraceData};
use crate::transport::http::config::ClientConfig;

/// Raw hostname lookup seam beneath transport-owned cache policy.
#[async_trait(?Send)]
pub trait HostLookup {
    /// Resolve `host:port` and return the preferred address.
    async fn lookup(&self, host: &str, port: u16) -> Result<SocketAddr, ErrorDetails>;
}

/// Tokio reactor-backed system hostname lookup.
#[derive(Debug, Default)]
pub struct TokioHostLookup;

#[async_trait(?Send)]
impl HostLookup for TokioHostLookup {
    async fn lookup(&self, host: &str, port: u16) -> Result<SocketAddr, ErrorDetails> {
        let mut addrs = tokio::net::lookup_host((host, port))
            .await
            .map_err(|error| ErrorDetails {
                kind: ErrorKind::Connect,
                code: None,
                message: format!("dns: {error}"),
            })?;
        addrs
            .next()
            .ok_or_else(|| ErrorDetails::other(format!("no address for {host}")))
    }
}

/// DNS policy seam used by connection establishment.
#[async_trait(?Send)]
pub trait DnsResolver {
    /// Resolve one endpoint while adding cache/lookup facts to `trace`.
    async fn resolve(
        &self,
        host: &str,
        port: u16,
        cfg: &ClientConfig,
        clock: &Rc<dyn Clock>,
        trace: &mut TraceData,
    ) -> Result<SocketAddr, ErrorDetails>;
}

#[derive(Debug, Clone, Copy)]
struct CachedAddress {
    address: SocketAddr,
    cached_at_ns: i64,
}

/// Per-transport DNS cache over an injected hostname lookup implementation.
pub struct CachingDnsResolver {
    lookup: Rc<dyn HostLookup>,
    entries: RefCell<HashMap<(String, u16), CachedAddress>>,
}

impl CachingDnsResolver {
    /// Build a cache over `lookup`.
    pub fn new(lookup: Rc<dyn HostLookup>) -> Self {
        Self {
            lookup,
            entries: RefCell::new(HashMap::new()),
        }
    }

    fn cached(&self, key: &(String, u16), now_ns: i64, ttl_ns: Option<i64>) -> Option<SocketAddr> {
        let cached = self.entries.borrow().get(key).copied()?;
        let expired = ttl_ns
            .filter(|ttl| *ttl >= 0)
            .is_some_and(|ttl| now_ns.saturating_sub(cached.cached_at_ns) >= ttl);
        if expired {
            self.entries.borrow_mut().remove(key);
            None
        } else {
            Some(cached.address)
        }
    }
}

impl Default for CachingDnsResolver {
    fn default() -> Self {
        Self::new(Rc::new(TokioHostLookup))
    }
}

#[async_trait(?Send)]
impl DnsResolver for CachingDnsResolver {
    async fn resolve(
        &self,
        host: &str,
        port: u16,
        cfg: &ClientConfig,
        clock: &Rc<dyn Clock>,
        trace: &mut TraceData,
    ) -> Result<SocketAddr, ErrorDetails> {
        let key = (host.to_ascii_lowercase(), port);
        let now_ns = clock.now_ns();
        if cfg.use_dns_cache
            && let Some(address) = self.cached(&key, now_ns, cfg.dns_cache_ttl_ns)
        {
            trace.dns_cache_hit_ns = Some(now_ns);
            return Ok(address);
        }

        trace.dns_cache_miss_ns = Some(now_ns);
        trace.dns_lookup_start_ns = Some(clock.now_ns());
        let address = self.lookup.lookup(host, port).await?;
        trace.dns_lookup_end_ns = Some(clock.now_ns());
        if cfg.use_dns_cache {
            self.entries.borrow_mut().insert(
                key,
                CachedAddress {
                    address,
                    cached_at_ns: clock.now_ns(),
                },
            );
        }
        Ok(address)
    }
}

/// Resolve through a request-local cache using default client policy.
pub async fn resolve(
    host: &str,
    port: u16,
    clock: &Rc<dyn Clock>,
    trace: &mut TraceData,
) -> Result<SocketAddr, ErrorDetails> {
    resolve_with_config(host, port, &ClientConfig::default(), clock, trace).await
}

/// Resolve through a request-local cache using explicit client policy.
///
/// Long-lived callers should inject one [`CachingDnsResolver`] through the
/// connection manager so later connections can observe cache hits.
pub async fn resolve_with_config(
    host: &str,
    port: u16,
    cfg: &ClientConfig,
    clock: &Rc<dyn Clock>,
    trace: &mut TraceData,
) -> Result<SocketAddr, ErrorDetails> {
    CachingDnsResolver::default()
        .resolve(host, port, cfg, clock, trace)
        .await
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use crate::clock::SimClock;

    use super::*;

    struct CountingLookup {
        calls: Cell<usize>,
        address: SocketAddr,
    }

    #[async_trait(?Send)]
    impl HostLookup for CountingLookup {
        async fn lookup(&self, _host: &str, _port: u16) -> Result<SocketAddr, ErrorDetails> {
            self.calls.set(self.calls.get() + 1);
            Ok(self.address)
        }
    }

    #[test]
    fn sim_clock_dns_cache_records_miss_hit_and_ttl_expiry() {
        let clock = Rc::new(SimClock::new());
        let clock_dyn: Rc<dyn Clock> = clock.clone();
        let lookup = Rc::new(CountingLookup {
            calls: Cell::new(0),
            address: "127.0.0.1:8080".parse().unwrap(),
        });
        let resolver = CachingDnsResolver::new(lookup.clone());
        let cfg = ClientConfig {
            dns_cache_ttl_ns: Some(10),
            ..ClientConfig::default()
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();

        let mut first = TraceData::default();
        runtime
            .block_on(resolver.resolve("example.test", 8080, &cfg, &clock_dyn, &mut first))
            .unwrap();
        assert_eq!(lookup.calls.get(), 1);
        assert_eq!(first.dns_cache_miss_ns, Some(0));
        assert_eq!(first.dns_cache_hit_ns, None);
        assert_eq!(first.dns_lookup_start_ns, Some(0));

        clock.advance_to(9);
        let mut second = TraceData::default();
        runtime
            .block_on(resolver.resolve("example.test", 8080, &cfg, &clock_dyn, &mut second))
            .unwrap();
        assert_eq!(lookup.calls.get(), 1);
        assert_eq!(second.dns_cache_hit_ns, Some(9));
        assert_eq!(second.dns_lookup_start_ns, None);

        clock.advance_to(10);
        let mut third = TraceData::default();
        runtime
            .block_on(resolver.resolve("example.test", 8080, &cfg, &clock_dyn, &mut third))
            .unwrap();
        assert_eq!(lookup.calls.get(), 2);
        assert_eq!(third.dns_cache_miss_ns, Some(10));
        assert_eq!(third.dns_cache_hit_ns, None);
    }

    #[test]
    fn disabling_cache_forces_lookup_and_miss_facts() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let lookup = Rc::new(CountingLookup {
            calls: Cell::new(0),
            address: "127.0.0.1:8080".parse().unwrap(),
        });
        let resolver = CachingDnsResolver::new(lookup.clone());
        let cfg = ClientConfig {
            use_dns_cache: false,
            ..ClientConfig::default()
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();

        for _ in 0..2 {
            let mut trace = TraceData::default();
            runtime
                .block_on(resolver.resolve("example.test", 8080, &cfg, &clock, &mut trace))
                .unwrap();
            assert_eq!(trace.dns_cache_miss_ns, Some(0));
            assert_eq!(trace.dns_cache_hit_ns, None);
        }
        assert_eq!(lookup.calls.get(), 2);
    }
}
