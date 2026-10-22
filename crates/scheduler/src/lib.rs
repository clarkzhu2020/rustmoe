//! Top-K preserving expert scheduling and cold-read coalescing.
//!
//! The router is authoritative. A step with fewer than the configured number
//! of experts, duplicated experts, or a missing layer is rejected. Bandwidth
//! pressure never changes model semantics.

use rustmoe_expert_cache::{ExpertKey, Tier, TieredExpertCache, TieredSnapshot};
use rustmoe_model::ModelDescriptor;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Clone, Debug, Serialize)]
pub struct ExpertChoice {
    pub expert: u32,
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayerRoute {
    pub layer: u16,
    pub experts: Vec<ExpertChoice>,
    pub shared_expert: bool,
}

impl LayerRoute {
    pub fn new(layer: u16, experts: impl IntoIterator<Item = (u32, f32)>) -> Self {
        Self {
            layer,
            experts: experts
                .into_iter()
                .map(|(expert, weight)| ExpertChoice { expert, weight })
                .collect(),
            shared_expert: true,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BlobGeometry {
    pub shard: usize,
    pub base_offset: u64,
    pub bytes_per_expert: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExpertGeometry {
    pub layer: usize,
    pub expert_count: u32,
    pub total_bytes: u64,
    pub load_cost: u64,
    pub gate: BlobGeometry,
    pub down: BlobGeometry,
    pub up: BlobGeometry,
}

#[derive(Clone, Debug, Serialize)]
pub struct SchedulerConfig {
    pub expected_layers: usize,
    pub expected_top_k: usize,
    pub expected_shared: usize,
    pub geometry: Vec<ExpertGeometry>,
    pub coalesce_gap: u64,
    pub prefetch_layers: usize,
}

impl SchedulerConfig {
    pub fn from_model(model: &ModelDescriptor, prefetch_layers: usize) -> Result<Self, SchedulerError> {
        let mut geometry = Vec::with_capacity(model.config.block_count);
        for layer in 0..model.config.block_count {
            let layout = model
                .expert_layout(layer)
                .ok_or(SchedulerError::MissingLayout(layer))?;
            if layout.expert_bytes == 0 {
                return Err(SchedulerError::MissingLayout(layer));
            }
            geometry.push(ExpertGeometry {
                layer,
                expert_count: layout.expert_count as u32,
                total_bytes: layout.expert_bytes,
                load_cost: layout.expert_bytes.saturating_add(256 * 1024),
                gate: BlobGeometry {
                    shard: layout.gate_shard,
                    base_offset: layout.gate_offset,
                    bytes_per_expert: layout.gate_bytes,
                },
                down: BlobGeometry {
                    shard: layout.down_shard,
                    base_offset: layout.down_offset,
                    bytes_per_expert: layout.down_bytes,
                },
                up: BlobGeometry {
                    shard: layout.up_shard,
                    base_offset: layout.up_offset,
                    bytes_per_expert: layout.up_bytes,
                },
            });
        }
        Ok(Self {
            expected_layers: model.config.block_count,
            expected_top_k: model.config.experts.used_count as usize,
            expected_shared: model.config.experts.shared_count as usize,
            geometry,
            coalesce_gap: 64 * 1024,
            prefetch_layers,
        })
    }

    /// Geometry for a simulation-only run when the full tensor catalog is not
    /// present. No bytes are read; this only exercises cache accounting.
    pub fn synthetic(
        layers: usize,
        top_k: usize,
        shared: usize,
        expert_count: u32,
        expert_bytes: u64,
        prefetch_layers: usize,
    ) -> Self {
        let geometry = (0..layers)
            .map(|layer| ExpertGeometry {
                layer,
                expert_count,
                total_bytes: expert_bytes,
                load_cost: expert_bytes,
                gate: BlobGeometry {
                    shard: 0,
                    base_offset: 1_000 + layer as u64 * 1_000_000,
                    bytes_per_expert: expert_bytes / 3,
                },
                down: BlobGeometry {
                    shard: 0,
                    base_offset: 500_000 + layer as u64 * 1_000_000,
                    bytes_per_expert: expert_bytes / 3,
                },
                up: BlobGeometry {
                    shard: 0,
                    base_offset: 900_000 + layer as u64 * 1_000_000,
                    bytes_per_expert: expert_bytes / 3,
                },
            })
            .collect();
        Self {
            expected_layers: layers,
            expected_top_k: top_k,
            expected_shared: shared,
            geometry,
            coalesce_gap: 16,
            prefetch_layers,
        }
    }

    pub fn for_tests(
        layers: usize,
        top_k: usize,
        shared: usize,
        expert_bytes: u64,
        prefetch_layers: usize,
    ) -> Self {
        Self::synthetic(layers, top_k, shared, 384, expert_bytes, prefetch_layers)
    }

    fn validate(&self) -> Result<(), SchedulerError> {
        if self.expected_layers == 0 || self.expected_top_k == 0 {
            return Err(SchedulerError::InvalidConfig);
        }
        if self.geometry.len() != self.expected_layers {
            return Err(SchedulerError::InvalidConfig);
        }
        if self.expected_layers > u16::MAX as usize + 1 {
            return Err(SchedulerError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ReadRange {
    pub shard: usize,
    pub offset: u64,
    pub len: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReadBatch {
    pub layer: usize,
    pub shard: usize,
    pub ranges: Vec<ReadRange>,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExpertPlan {
    pub key: ExpertKey,
    pub bytes: u64,
    pub tier: Tier,
    pub prefetched: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayerStepPlan {
    pub layer: usize,
    pub experts: Vec<ExpertPlan>,
    pub gpu_hits: u64,
    pub host_hits: u64,
    pub disk_misses: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct StepPlan {
    pub layers: Vec<LayerStepPlan>,
    pub requests: u64,
    pub gpu_hits: u64,
    pub host_hits: u64,
    pub disk_misses: u64,
    pub disk_bytes: u64,
    pub h2d_bytes: u64,
    pub prefetch_issued: u64,
    pub prefetch_bytes: u64,
    pub prefetch_hits: u64,
    pub read_batches: Vec<ReadBatch>,
    pub coalesced_bytes: u64,
    pub evictions: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SchedulerStats {
    pub steps: u64,
    pub requests: u64,
    pub gpu_hits: u64,
    pub host_hits: u64,
    pub disk_misses: u64,
    pub disk_bytes: u64,
    pub h2d_bytes: u64,
    pub read_batches: u64,
    pub coalesced_bytes: u64,
    pub prefetch_issued: u64,
    pub prefetch_bytes: u64,
    pub prefetch_hits: u64,
    pub evictions: u64,
}

impl SchedulerStats {
    pub fn hit_rate(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            (self.gpu_hits + self.host_hits) as f64 / self.requests as f64
        }
    }

    fn add(&mut self, plan: &StepPlan) {
        self.steps += 1;
        self.requests += plan.requests;
        self.gpu_hits += plan.gpu_hits;
        self.host_hits += plan.host_hits;
        self.disk_misses += plan.disk_misses;
        self.disk_bytes = self.disk_bytes.saturating_add(plan.disk_bytes);
        self.h2d_bytes = self.h2d_bytes.saturating_add(plan.h2d_bytes);
        self.read_batches += plan.read_batches.len() as u64;
        self.coalesced_bytes = self.coalesced_bytes.saturating_add(plan.coalesced_bytes);
        self.prefetch_issued += plan.prefetch_issued;
        self.prefetch_bytes = self.prefetch_bytes.saturating_add(plan.prefetch_bytes);
        self.prefetch_hits += plan.prefetch_hits;
        self.evictions += plan.evictions;
    }
}

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("invalid scheduler configuration")]
    InvalidConfig,
    #[error("missing expert layout for layer {0}")]
    MissingLayout(usize),
    #[error("step has {actual} layers, expected {expected}")]
    LayerCount { expected: usize, actual: usize },
    #[error("layer {0} is duplicated")]
    DuplicateLayer(u16),
    #[error("layer {layer} has {actual} experts, expected {expected}; refusing to change Top-K")]
    TopK {
        layer: u16,
        expected: usize,
        actual: usize,
    },
    #[error("layer {layer} has duplicate expert {expert}")]
    DuplicateExpert { layer: u16, expert: u32 },
    #[error("layer {layer} expert {expert} is out of range")]
    ExpertOutOfRange { layer: u16, expert: u32 },
    #[error("layer {0} is outside the model")]
    LayerOutOfRange(u16),
    #[error("layer {0} did not mark its shared expert as computed")]
    SharedExpertMissing(u16),
    #[error("layer {layer} expert {expert} has a non-finite weight")]
    NonFiniteWeight { layer: u16, expert: u32 },
    #[error("layer {0} has no geometry")]
    NoGeometry(u16),
}

#[derive(Clone, Debug)]
pub struct ExpertScheduler {
    config: SchedulerConfig,
    cache: TieredExpertCache,
    stats: SchedulerStats,
    prefetched: BTreeSet<ExpertKey>,
}

impl ExpertScheduler {
    pub fn new(config: SchedulerConfig, gpu_bytes: u64, host_bytes: u64) -> Result<Self, SchedulerError> {
        config.validate()?;
        Ok(Self {
            cache: TieredExpertCache::new(gpu_bytes, host_bytes),
            config,
            stats: SchedulerStats::default(),
            prefetched: BTreeSet::new(),
        })
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    pub fn cache(&self) -> &TieredExpertCache {
        &self.cache
    }

    pub fn cache_snapshot(&self) -> TieredSnapshot {
        self.cache.snapshot()
    }

    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = SchedulerStats::default();
        self.cache.reset_stats();
        self.prefetched.clear();
    }

    /// Plan one decode step. A prefetch prediction for the next step may warm
    /// the host tier, but it can never change the current step expert set.
    pub fn plan_step(
        &mut self,
        routes: &[LayerRoute],
        prefetch: Option<&[LayerRoute]>,
    ) -> Result<StepPlan, SchedulerError> {
        self.validate_routes(routes)?;
        if let Some(predicted) = prefetch {
            self.validate_routes(predicted)?;
        }
        let evictions_before = self.cache.snapshot();
        let mut plan = StepPlan::default();
        let mut misses: Vec<(usize, u32, ExpertGeometry)> = Vec::new();
        let mut layers: Vec<LayerStepPlan> = Vec::with_capacity(routes.len());

        for route in routes.iter().cloned() {
            let layer = route.layer as usize;
            let geometry = self
                .config
                .geometry
                .get(layer)
                .cloned()
                .ok_or(SchedulerError::NoGeometry(route.layer))?;
            let mut experts = Vec::with_capacity(route.experts.len());
            let mut layer_gpu = 0;
            let mut layer_host = 0;
            let mut layer_disk = 0;
            let mut seen = BTreeSet::new();
            for choice in route.experts {
                if !choice.weight.is_finite() {
                    return Err(SchedulerError::NonFiniteWeight {
                        layer: route.layer,
                        expert: choice.expert,
                    });
                }
                if !seen.insert(choice.expert) {
                    return Err(SchedulerError::DuplicateExpert {
                        layer: route.layer,
                        expert: choice.expert,
                    });
                }
                let key = ExpertKey::new(layer, choice.expert);
                let was_prefetched = self.prefetched.contains(&key);
                let tier = self.cache.resolve(key, geometry.total_bytes, geometry.load_cost);
                match tier {
                    Tier::Gpu => {
                        layer_gpu += 1;
                        plan.gpu_hits += 1;
                    }
                    Tier::Host => {
                        layer_host += 1;
                        plan.host_hits += 1;
                        plan.h2d_bytes = plan.h2d_bytes.saturating_add(geometry.total_bytes);
                        if was_prefetched {
                            plan.prefetch_hits += 1;
                            self.prefetched.remove(&key);
                        }
                    }
                    Tier::Disk => {
                        layer_disk += 1;
                        plan.disk_misses += 1;
                        plan.disk_bytes = plan.disk_bytes.saturating_add(geometry.total_bytes);
                        plan.h2d_bytes = plan.h2d_bytes.saturating_add(geometry.total_bytes);
                        misses.push((layer, choice.expert, geometry.clone()));
                        self.cache
                            .note_loaded(key, geometry.total_bytes, geometry.load_cost, true);
                    }
                }
                experts.push(ExpertPlan {
                    key,
                    bytes: geometry.total_bytes,
                    tier,
                    prefetched: was_prefetched,
                });
            }
            plan.requests += experts.len() as u64;
            layers.push(LayerStepPlan {
                layer,
                experts,
                gpu_hits: layer_gpu,
                host_hits: layer_host,
                disk_misses: layer_disk,
            });
        }

        // Last step prefetch predictions expire after this demand pass.
        self.prefetched.clear();

        plan.read_batches = self.build_read_batches(&misses);
        plan.coalesced_bytes = plan.read_batches.iter().map(|b| b.bytes).sum();

        if let Some(predicted) = prefetch {
            if self.config.prefetch_layers > 0 {
                self.issue_prefetch(predicted, &mut plan)?;
            }
        }

        let evictions_after = self.cache.snapshot();
        plan.evictions = (evictions_after.gpu.evictions + evictions_after.host.evictions)
            .saturating_sub(evictions_before.gpu.evictions + evictions_before.host.evictions);
        self.stats.add(&plan);
        Ok(plan)
    }

    fn issue_prefetch(&mut self, routes: &[LayerRoute], plan: &mut StepPlan) -> Result<(), SchedulerError> {
        let limit = routes.len().min(self.config.prefetch_layers);
        let mut seen = BTreeSet::new();
        for route in &routes[..limit] {
            let layer = route.layer as usize;
            if layer >= self.config.expected_layers {
                continue;
            }
            let geometry = match self.config.geometry.get(layer) {
                Some(g) => g,
                None => continue,
            };
            for choice in &route.experts {
                let key = ExpertKey::new(layer, choice.expert);
                if !seen.insert(key) || self.cache.gpu.contains(key) || self.cache.host.contains(key) {
                    continue;
                }
                self.cache
                    .note_loaded(key, geometry.total_bytes, geometry.load_cost, false);
                self.prefetched.insert(key);
                plan.prefetch_issued += 1;
                plan.prefetch_bytes = plan.prefetch_bytes.saturating_add(geometry.total_bytes);
            }
        }
        Ok(())
    }

    fn validate_routes(&self, routes: &[LayerRoute]) -> Result<(), SchedulerError> {
        if routes.len() != self.config.expected_layers {
            return Err(SchedulerError::LayerCount {
                expected: self.config.expected_layers,
                actual: routes.len(),
            });
        }
        let mut layers = BTreeSet::new();
        for route in routes {
            let layer = route.layer as usize;
            if layer >= self.config.expected_layers {
                return Err(SchedulerError::LayerOutOfRange(route.layer));
            }
            if !layers.insert(route.layer) {
                return Err(SchedulerError::DuplicateLayer(route.layer));
            }
            if route.experts.len() != self.config.expected_top_k {
                return Err(SchedulerError::TopK {
                    layer: route.layer,
                    expected: self.config.expected_top_k,
                    actual: route.experts.len(),
                });
            }
            if self.config.expected_shared > 0 && !route.shared_expert {
                return Err(SchedulerError::SharedExpertMissing(route.layer));
            }
            let geometry = self
                .config
                .geometry
                .get(layer)
                .ok_or(SchedulerError::NoGeometry(route.layer))?;
            for choice in &route.experts {
                if !choice.weight.is_finite() {
                    return Err(SchedulerError::NonFiniteWeight {
                        layer: route.layer,
                        expert: choice.expert,
                    });
                }
                if choice.expert >= geometry.expert_count {
                    return Err(SchedulerError::ExpertOutOfRange {
                        layer: route.layer,
                        expert: choice.expert,
                    });
                }
            }
        }
        Ok(())
    }

    fn build_read_batches(&self, misses: &[(usize, u32, ExpertGeometry)]) -> Vec<ReadBatch> {
        #[derive(Clone)]
        struct PendingRange {
            layer: usize,
            shard: usize,
            offset: u64,
            len: u64,
        }
        let mut pending = Vec::new();
        for (layer, expert, geometry) in misses {
            for blob in [&geometry.gate, &geometry.down, &geometry.up] {
                let offset = blob.base_offset + *expert as u64 * blob.bytes_per_expert;
                pending.push(PendingRange {
                    layer: *layer,
                    shard: blob.shard,
                    offset,
                    len: blob.bytes_per_expert,
                });
            }
        }
        pending.sort_by_key(|r| (r.layer, r.shard, r.offset));
        let mut batches: Vec<ReadBatch> = Vec::new();
        for range in pending {
            if range.len == 0 {
                continue;
            }
            if let Some(batch) = batches.last_mut() {
                if batch.layer == range.layer && batch.shard == range.shard {
                    if let Some(last) = batch.ranges.last_mut() {
                        let end = last.offset.saturating_add(last.len);
                        if range.offset <= end.saturating_add(self.config.coalesce_gap)
                            && range.offset >= last.offset
                        {
                            let new_end = end.max(range.offset.saturating_add(range.len));
                            last.len = new_end - last.offset;
                            batch.bytes = batch.bytes.saturating_add(range.len);
                            continue;
                        }
                    }
                }
            }
            batches.push(ReadBatch {
                layer: range.layer,
                shard: range.shard,
                ranges: vec![ReadRange {
                    shard: range.shard,
                    offset: range.offset,
                    len: range.len,
                }],
                bytes: range.len,
            });
        }
        for b in &mut batches {
            b.bytes = b.ranges.iter().map(|r| r.len).fold(0u64, u64::saturating_add);
        }
        batches
    }
}

/// Group routes by layer for backends that receive one layer at a time.
pub fn routes_by_layer(routes: &[LayerRoute]) -> BTreeMap<u16, &LayerRoute> {
    routes.iter().map(|r| (r.layer, r)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routes(layers: usize, top_k: usize, seed: u32) -> Vec<LayerRoute> {
        (0..layers as u16)
            .map(|layer| {
                LayerRoute::new(
                    layer,
                    (0..top_k).map(|i| (seed + layer as u32 * 10 + i as u32, 0.5)),
                )
            })
            .collect()
    }

    #[test]
    fn cold_step_reads_every_expert() {
        let cfg = SchedulerConfig::for_tests(4, 2, 1, 120, 0);
        let mut s = ExpertScheduler::new(cfg, 240, 480).unwrap();
        let plan = s.plan_step(&routes(4, 2, 0), None).unwrap();
        assert_eq!(plan.requests, 8);
        assert_eq!(plan.disk_misses, 8);
        assert_eq!(plan.disk_bytes, 960);
        assert!(plan.host_hits == 0 && plan.gpu_hits == 0);
        assert!(!plan.read_batches.is_empty());
    }

    #[test]
    fn second_step_uses_cache() {
        let cfg = SchedulerConfig::for_tests(4, 2, 1, 120, 0);
        let mut s = ExpertScheduler::new(cfg, 480, 960).unwrap();
        let r = routes(4, 2, 0);
        let first = s.plan_step(&r, None).unwrap();
        assert_eq!(first.disk_misses, 8);
        let second = s.plan_step(&r, None).unwrap();
        assert_eq!(second.disk_misses, 0);
        assert_eq!(second.gpu_hits + second.host_hits, 8);
        assert!(second.h2d_bytes > 0);
    }

    #[test]
    fn top_k_is_not_negotiable() {
        let cfg = SchedulerConfig::for_tests(1, 2, 1, 120, 0);
        let mut s = ExpertScheduler::new(cfg, 1000, 1000).unwrap();
        let bad = vec![LayerRoute::new(0, [(1, 1.0)])];
        let err = s.plan_step(&bad, None).unwrap_err();
        assert!(matches!(err, SchedulerError::TopK { .. }));

        let dup = vec![LayerRoute::new(0, [(1, 1.0), (1, 1.0)])];
        let err = s.plan_step(&dup, None).unwrap_err();
        assert!(matches!(err, SchedulerError::DuplicateExpert { .. }));
    }

    #[test]
    fn missing_layer_is_rejected() {
        let cfg = SchedulerConfig::for_tests(2, 2, 1, 120, 0);
        let mut s = ExpertScheduler::new(cfg, 1000, 1000).unwrap();
        let one = vec![LayerRoute::new(0, [(1, 1.0), (2, 1.0)])];
        assert!(matches!(
            s.plan_step(&one, None).unwrap_err(),
            SchedulerError::LayerCount { .. }
        ));
    }

    #[test]
    fn prefetch_records_next_step_hits() {
        let cfg = SchedulerConfig::for_tests(2, 2, 1, 120, 2);
        let mut s = ExpertScheduler::new(cfg, 0, 960).unwrap();
        let first_routes = routes(2, 2, 0);
        let next_routes = routes(2, 2, 100);
        let first = s.plan_step(&first_routes, Some(&next_routes)).unwrap();
        assert_eq!(first.disk_misses, 4);
        assert_eq!(first.prefetch_issued, 4);
        let second = s.plan_step(&next_routes, None).unwrap();
        assert_eq!(second.disk_misses, 0);
        assert_eq!(second.prefetch_hits, 4);
    }

    #[test]
    fn batch_coalesces_adjacent_experts() {
        let cfg = SchedulerConfig::for_tests(1, 2, 1, 120, 0);
        let mut s = ExpertScheduler::new(cfg, 0, 0).unwrap();
        let r = vec![LayerRoute::new(0, [(0, 1.0), (1, 1.0)])];
        let plan = s.plan_step(&r, None).unwrap();
        assert!(plan.read_batches.len() <= 3);
        assert!(plan.coalesced_bytes >= plan.disk_bytes);
    }
}
