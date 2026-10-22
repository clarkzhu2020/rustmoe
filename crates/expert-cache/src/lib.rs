//! Cost-aware expert caches.
//!
//! The cache owns bookkeeping, not GPU memory. A backend allocates the actual
//! buffers and asks this module which experts should remain resident. The host
//! tier uses the same policy but a larger budget.

use serde::Serialize;
use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct ExpertKey {
    pub layer: u16,
    pub expert: u16,
}

impl ExpertKey {
    pub fn new(layer: usize, expert: u32) -> Self {
        Self {
            layer: layer as u16,
            expert: expert as u16,
        }
    }
}

impl fmt::Display for ExpertKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}E{}", self.layer, self.expert)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionPolicy {
    Lru,
    Lfu,
    CostAware,
}

#[derive(Clone, Debug, Serialize)]
pub struct CacheConfig {
    pub capacity_bytes: u64,
    pub policy: EvictionPolicy,
    pub frequency_weight: f64,
    pub recency_weight: f64,
    pub cost_weight: f64,
}

impl CacheConfig {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            policy: EvictionPolicy::CostAware,
            frequency_weight: 1.0,
            recency_weight: 2.0,
            cost_weight: 1.0,
        }
    }

    pub fn lru(capacity_bytes: u64) -> Self {
        Self {
            policy: EvictionPolicy::Lru,
            ..Self::new(capacity_bytes)
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CacheStats {
    pub capacity_bytes: u64,
    pub current_bytes: u64,
    pub entries: u64,
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub updates: u64,
    pub evictions: u64,
    pub bytes_evicted: u64,
    pub bypasses: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.lookups as f64
        }
    }
}

#[derive(Clone, Debug)]
struct Entry {
    bytes: u64,
    hits: u64,
    last_access: u64,
    inserted_at: u64,
    load_cost: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InsertOutcome {
    Inserted,
    Updated,
    Bypassed,
}

#[derive(Clone, Debug, Serialize)]
pub struct CacheEntrySnapshot {
    pub key: ExpertKey,
    pub bytes: u64,
    pub hits: u64,
    pub last_access: u64,
    pub inserted_at: u64,
    pub load_cost: u64,
}

#[derive(Clone, Debug)]
pub struct ExpertCache {
    config: CacheConfig,
    entries: HashMap<ExpertKey, Entry>,
    current_bytes: u64,
    clock: u64,
    stats: CacheStats,
}

impl ExpertCache {
    pub fn new(config: CacheConfig) -> Self {
        let stats = CacheStats {
            capacity_bytes: config.capacity_bytes,
            ..CacheStats::default()
        };
        Self {
            config,
            entries: HashMap::new(),
            current_bytes: 0,
            clock: 0,
            stats,
        }
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.config.capacity_bytes
    }

    pub fn current_bytes(&self) -> u64 {
        self.current_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, key: ExpertKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Look up and touch an entry. This updates frequency and recency.
    pub fn get(&mut self, key: ExpertKey) -> bool {
        self.clock += 1;
        self.stats.lookups += 1;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.hits = entry.hits.saturating_add(1);
            entry.last_access = self.clock;
            self.stats.hits += 1;
            true
        } else {
            self.stats.misses += 1;
            false
        }
    }

    pub fn insert(&mut self, key: ExpertKey, bytes: u64, load_cost: u64) -> InsertOutcome {
        self.clock += 1;
        if bytes == 0 {
            return InsertOutcome::Bypassed;
        }
        if bytes > self.config.capacity_bytes {
            self.stats.bypasses += 1;
            return InsertOutcome::Bypassed;
        }
        if let Some(entry) = self.entries.get_mut(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(entry.bytes);
            entry.bytes = bytes;
            entry.last_access = self.clock;
            entry.load_cost = load_cost;
            self.current_bytes = self.current_bytes.saturating_add(bytes);
            self.stats.updates += 1;
            return InsertOutcome::Updated;
        }

        while self.current_bytes.saturating_add(bytes) > self.config.capacity_bytes {
            if self.evict_one().is_none() {
                self.stats.bypasses += 1;
                return InsertOutcome::Bypassed;
            }
        }
        self.entries.insert(
            key,
            Entry {
                bytes,
                hits: 0,
                last_access: self.clock,
                inserted_at: self.clock,
                load_cost,
            },
        );
        self.current_bytes = self.current_bytes.saturating_add(bytes);
        self.stats.insertions += 1;
        InsertOutcome::Inserted
    }

    pub fn remove(&mut self, key: ExpertKey) -> bool {
        if let Some(entry) = self.entries.remove(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(entry.bytes);
            true
        } else {
            false
        }
    }

    fn evict_one(&mut self) -> Option<ExpertKey> {
        let victim = self
            .entries
            .iter()
            .min_by(|(ka, a), (kb, b)| {
                self.victim_score(a)
                    .partial_cmp(&self.victim_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| kb.cmp(ka))
            })
            .map(|(k, _)| *k)?;
        let entry = self.entries.remove(&victim)?;
        self.current_bytes = self.current_bytes.saturating_sub(entry.bytes);
        self.stats.evictions += 1;
        self.stats.bytes_evicted = self.stats.bytes_evicted.saturating_add(entry.bytes);
        Some(victim)
    }

    fn victim_score(&self, entry: &Entry) -> f64 {
        match self.config.policy {
            EvictionPolicy::Lru => entry.last_access as f64,
            EvictionPolicy::Lfu => entry.hits as f64 + entry.last_access as f64 * 1e-9,
            EvictionPolicy::CostAware => {
                let age = self.clock.saturating_sub(entry.last_access) as f64;
                let recency = 1.0 / (1.0 + age / 32.0);
                let frequency = (1.0 + entry.hits as f64).ln();
                let cost = (1.0 + entry.load_cost as f64).ln() / 8.0;
                let value = self.config.frequency_weight * frequency
                    + self.config.recency_weight * recency
                    + self.config.cost_weight * cost;
                value / (entry.bytes as f64).sqrt().max(1.0)
            }
        }
    }

    pub fn stats(&self) -> CacheStats {
        let mut stats = self.stats.clone();
        stats.current_bytes = self.current_bytes;
        stats.entries = self.entries.len() as u64;
        stats.capacity_bytes = self.config.capacity_bytes;
        stats
    }

    pub fn snapshot(&self) -> Vec<CacheEntrySnapshot> {
        let mut out: Vec<_> = self
            .entries
            .iter()
            .map(|(key, e)| CacheEntrySnapshot {
                key: *key,
                bytes: e.bytes,
                hits: e.hits,
                last_access: e.last_access,
                inserted_at: e.inserted_at,
                load_cost: e.load_cost,
            })
            .collect();
        out.sort_by_key(|e| e.key);
        out
    }

    pub fn reset_stats(&mut self) {
        self.stats = CacheStats {
            capacity_bytes: self.config.capacity_bytes,
            ..CacheStats::default()
        };
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Gpu,
    Host,
    Disk,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TieredStats {
    pub gpu_hits: u64,
    pub host_hits: u64,
    pub disk_misses: u64,
    pub tokens_observed: u64,
    pub expert_requests: u64,
}

impl TieredStats {
    pub fn hit_rate(&self) -> f64 {
        if self.expert_requests == 0 {
            0.0
        } else {
            (self.gpu_hits + self.host_hits) as f64 / self.expert_requests as f64
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TieredSnapshot {
    pub stats: TieredStats,
    pub gpu: CacheStats,
    pub host: CacheStats,
}

/// Two-level cache bookkeeping. A backend owns the actual GPU/host buffers;
/// this type decides residency and records hit/miss behavior.
#[derive(Clone, Debug)]
pub struct TieredExpertCache {
    pub gpu: ExpertCache,
    pub host: ExpertCache,
    stats: TieredStats,
}

impl TieredExpertCache {
    pub fn new(gpu_bytes: u64, host_bytes: u64) -> Self {
        Self {
            gpu: ExpertCache::new(CacheConfig::new(gpu_bytes)),
            host: ExpertCache::new(CacheConfig::new(host_bytes)),
            stats: TieredStats::default(),
        }
    }

    /// Resolve an expert request without inserting anything. A Disk result
    /// means the caller must read the cold store and then call note_loaded.
    pub fn resolve(&mut self, key: ExpertKey, bytes: u64, load_cost: u64) -> Tier {
        self.stats.expert_requests += 1;
        if self.gpu.get(key) {
            self.stats.gpu_hits += 1;
            return Tier::Gpu;
        }
        if self.host.get(key) {
            self.stats.host_hits += 1;
            // Promote to GPU. The backend performs the actual H2D copy.
            let _ = self.gpu.insert(key, bytes, load_cost);
            return Tier::Host;
        }
        self.stats.disk_misses += 1;
        Tier::Disk
    }

    /// Account for a load that has reached host memory. The caller may also
    /// place the same bytes in the GPU tier.
    pub fn note_loaded(&mut self, key: ExpertKey, bytes: u64, load_cost: u64, to_gpu: bool) {
        let _ = self.host.insert(key, bytes, load_cost);
        if to_gpu {
            let _ = self.gpu.insert(key, bytes, load_cost);
        }
    }

    /// Record one decode/prefill step. Used for average experts per token.
    pub fn note_step(&mut self, experts: u64) {
        self.stats.tokens_observed += 1;
        self.stats.expert_requests = self.stats.expert_requests.saturating_add(0);
        let _ = experts;
    }

    pub fn stats(&self) -> &TieredStats {
        &self.stats
    }

    pub fn snapshot(&self) -> TieredSnapshot {
        TieredSnapshot {
            stats: self.stats.clone(),
            gpu: self.gpu.stats(),
            host: self.host.stats(),
        }
    }

    pub fn reset_stats(&mut self) {
        self.gpu.reset_stats();
        self.host.reset_stats();
        self.stats = TieredStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(layer: usize, expert: u32) -> ExpertKey {
        ExpertKey::new(layer, expert)
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let mut c = ExpertCache::new(CacheConfig::lru(100));
        c.insert(key(0, 1), 50, 1);
        c.insert(key(0, 2), 50, 1);
        assert!(c.get(key(0, 1)));
        c.insert(key(0, 3), 50, 1);
        assert!(c.contains(key(0, 1)));
        assert!(!c.contains(key(0, 2)));
        assert!(c.contains(key(0, 3)));
    }

    #[test]
    fn cost_aware_keeps_expensive_hot_expert() {
        let mut c = ExpertCache::new(CacheConfig::new(100));
        c.insert(key(0, 1), 50, 1);
        c.insert(key(0, 2), 50, 1_000_000);
        for _ in 0..5 {
            assert!(c.get(key(0, 2)));
        }
        c.insert(key(0, 3), 50, 1);
        assert!(!c.contains(key(0, 1)));
        assert!(c.contains(key(0, 2)));
        assert!(c.contains(key(0, 3)));
    }

    #[test]
    fn oversized_entry_bypasses() {
        let mut c = ExpertCache::new(CacheConfig::new(100));
        assert_eq!(c.insert(key(0, 1), 101, 1), InsertOutcome::Bypassed);
        assert!(c.is_empty());
        assert_eq!(c.stats().bypasses, 1);
    }

    #[test]
    fn tiered_resolution_promotes_and_falls_back() {
        let mut c = TieredExpertCache::new(100, 200);
        let k = key(1, 7);
        assert_eq!(c.resolve(k, 100, 1), Tier::Disk);
        c.note_loaded(k, 100, 1, true);
        assert_eq!(c.resolve(k, 100, 1), Tier::Gpu);

        // Fill the GPU tier with another expert; the old one remains in host.
        let other = key(1, 8);
        assert_eq!(c.resolve(other, 100, 1), Tier::Disk);
        c.note_loaded(other, 100, 1, true);
        assert_eq!(c.resolve(k, 100, 1), Tier::Host);
        let snap = c.snapshot();
        assert!(snap.stats.gpu_hits >= 1);
        assert!(snap.stats.host_hits >= 1);
        assert!(snap.stats.hit_rate() > 0.0);
    }

    #[test]
    fn stats_track_hit_rate() {
        let mut c = ExpertCache::new(CacheConfig::new(100));
        c.insert(key(0, 1), 100, 1);
        assert!(c.get(key(0, 1)));
        assert!(!c.get(key(0, 2)));
        let s = c.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert_eq!(s.hit_rate(), 0.5);
    }
}
