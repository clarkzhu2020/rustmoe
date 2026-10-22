//! Runtime counters and latency summaries with Prometheus exposition.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

const LATENCY_BUCKETS_MS: [f64; 14] = [
    1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10_000.0, 30_000.0,
];

#[derive(Debug, Default)]
struct Latency {
    count: AtomicU64,
    sum_ns: AtomicU64,
    max_ns: AtomicU64,
    buckets: [AtomicU64; LATENCY_BUCKETS_MS.len()],
}

impl Latency {
    fn observe(&self, d: Duration) {
        let ns = d.as_nanos().min(u64::MAX as u128) as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(ns, Ordering::Relaxed);
        self.max_ns.fetch_max(ns, Ordering::Relaxed);
        let ms = d.as_secs_f64() * 1000.0;
        for (i, b) in LATENCY_BUCKETS_MS.iter().enumerate() {
            if ms <= *b {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    fn snapshot(&self) -> LatencySnapshot {
        let count = self.count.load(Ordering::Relaxed);
        let sum = self.sum_ns.load(Ordering::Relaxed);
        let max = self.max_ns.load(Ordering::Relaxed);
        let buckets = LATENCY_BUCKETS_MS
            .iter()
            .zip(self.buckets.iter())
            .map(|(b, c)| (*b, c.load(Ordering::Relaxed)))
            .collect();
        LatencySnapshot {
            count,
            sum_ms: sum as f64 / 1_000_000.0,
            mean_ms: if count == 0 {
                0.0
            } else {
                sum as f64 / count as f64 / 1_000_000.0
            },
            max_ms: max as f64 / 1_000_000.0,
            buckets,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LatencySnapshot {
    pub count: u64,
    pub sum_ms: f64,
    pub mean_ms: f64,
    pub max_ms: f64,
    pub buckets: Vec<(f64, u64)>,
}

impl LatencySnapshot {
    /// Approximate quantile from the upper bucket edge. This is intentionally
    /// coarse; use the raw timings in a benchmark when precision matters.
    pub fn quantile_ms(&self, q: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let target = (self.count as f64 * q.clamp(0.0, 1.0)).ceil() as u64;
        let mut cumulative = 0;
        for (edge, count) in &self.buckets {
            cumulative += *count;
            if cumulative >= target {
                return *edge;
            }
        }
        self.max_ms
    }
}

#[derive(Debug, Default)]
struct Counters {
    requests: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    prefill_chunks: AtomicU64,
    decode_steps: AtomicU64,
    errors: AtomicU64,
    gpu_cache_hits: AtomicU64,
    host_cache_hits: AtomicU64,
    disk_misses: AtomicU64,
    disk_bytes: AtomicU64,
    h2d_bytes: AtomicU64,
    cache_evictions: AtomicU64,
    prefetch_issued: AtomicU64,
    prefetch_hits: AtomicU64,
    active_requests: AtomicU64,
    context_tokens: AtomicU64,
    backend_cache_present: AtomicU64,
    backend_cache_hits: AtomicU64,
    backend_cache_misses: AtomicU64,
    backend_cache_uploaded_bytes: AtomicU64,
    backend_cache_copied_hits: AtomicU64,
    backend_cache_copied_bytes: AtomicU64,
}

#[derive(Debug, Default)]
struct Inner {
    counters: Counters,
    ttft: Latency,
    prefill: Latency,
    decode: Latency,
}

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    inner: Arc<Inner>,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn request_started(&self) {
        self.inner.counters.requests.fetch_add(1, Ordering::Relaxed);
        self.inner
            .counters
            .active_requests
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn request_finished(&self) {
        self.inner
            .counters
            .active_requests
            .fetch_sub(1, Ordering::Relaxed);
    }

    pub fn error(&self) {
        self.inner.counters.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_prompt_tokens(&self, n: u64) {
        self.inner.counters.prompt_tokens.fetch_add(n, Ordering::Relaxed);
    }

    pub fn add_completion_tokens(&self, n: u64) {
        self.inner
            .counters
            .completion_tokens
            .fetch_add(n, Ordering::Relaxed);
    }

    pub fn add_prefill_chunk(&self) {
        self.inner.counters.prefill_chunks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_decode_step(&self) {
        self.inner.counters.decode_steps.fetch_add(1, Ordering::Relaxed);
    }

    pub fn observe_ttft(&self, d: Duration) {
        self.inner.ttft.observe(d);
    }

    pub fn observe_prefill(&self, d: Duration) {
        self.inner.prefill.observe(d);
    }

    pub fn observe_decode(&self, d: Duration) {
        self.inner.decode.observe(d);
    }

    pub fn add_scheduler_step(&self, gpu: u64, host: u64, disk: u64, disk_bytes: u64, h2d: u64) {
        let c = &self.inner.counters;
        c.gpu_cache_hits.fetch_add(gpu, Ordering::Relaxed);
        c.host_cache_hits.fetch_add(host, Ordering::Relaxed);
        c.disk_misses.fetch_add(disk, Ordering::Relaxed);
        c.disk_bytes.fetch_add(disk_bytes, Ordering::Relaxed);
        c.h2d_bytes.fetch_add(h2d, Ordering::Relaxed);
    }

    pub fn add_prefetch(&self, issued: u64, hits: u64) {
        self.inner
            .counters
            .prefetch_issued
            .fetch_add(issued, Ordering::Relaxed);
        self.inner
            .counters
            .prefetch_hits
            .fetch_add(hits, Ordering::Relaxed);
    }

    pub fn add_evictions(&self, n: u64) {
        self.inner
            .counters
            .cache_evictions
            .fetch_add(n, Ordering::Relaxed);
    }

    pub fn set_context_tokens(&self, n: u64) {
        self.inner.counters.context_tokens.store(n, Ordering::Relaxed);
    }

    pub fn set_backend_cache(
        &self,
        hits: u64,
        misses: u64,
        uploaded_bytes: u64,
        copied_hits: u64,
        copied_bytes: u64,
    ) {
        let c = &self.inner.counters;
        c.backend_cache_present.store(1, Ordering::Relaxed);
        c.backend_cache_hits.store(hits, Ordering::Relaxed);
        c.backend_cache_misses.store(misses, Ordering::Relaxed);
        c.backend_cache_uploaded_bytes
            .store(uploaded_bytes, Ordering::Relaxed);
        c.backend_cache_copied_hits.store(copied_hits, Ordering::Relaxed);
        c.backend_cache_copied_bytes
            .store(copied_bytes, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let c = &self.inner.counters;
        let gpu = c.gpu_cache_hits.load(Ordering::Relaxed);
        let host = c.host_cache_hits.load(Ordering::Relaxed);
        let miss = c.disk_misses.load(Ordering::Relaxed);
        let requests = c.requests.load(Ordering::Relaxed);
        let backend_cache = if c.backend_cache_present.load(Ordering::Relaxed) != 0 {
            let hits = c.backend_cache_hits.load(Ordering::Relaxed);
            let misses = c.backend_cache_misses.load(Ordering::Relaxed);
            Some(BackendCacheMetrics {
                hits,
                misses,
                hit_rate: if hits + misses == 0 {
                    0.0
                } else {
                    hits as f64 / (hits + misses) as f64
                },
                uploaded_bytes: c.backend_cache_uploaded_bytes.load(Ordering::Relaxed),
                copied_hits: c.backend_cache_copied_hits.load(Ordering::Relaxed),
                copied_bytes: c.backend_cache_copied_bytes.load(Ordering::Relaxed),
            })
        } else {
            None
        };
        MetricsSnapshot {
            requests,
            active_requests: c.active_requests.load(Ordering::Relaxed),
            errors: c.errors.load(Ordering::Relaxed),
            prompt_tokens: c.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: c.completion_tokens.load(Ordering::Relaxed),
            prefill_chunks: c.prefill_chunks.load(Ordering::Relaxed),
            decode_steps: c.decode_steps.load(Ordering::Relaxed),
            gpu_cache_hits: gpu,
            host_cache_hits: host,
            disk_misses: miss,
            cache_hit_rate: if gpu + host + miss == 0 {
                0.0
            } else {
                (gpu + host) as f64 / (gpu + host + miss) as f64
            },
            disk_bytes: c.disk_bytes.load(Ordering::Relaxed),
            h2d_bytes: c.h2d_bytes.load(Ordering::Relaxed),
            cache_evictions: c.cache_evictions.load(Ordering::Relaxed),
            prefetch_issued: c.prefetch_issued.load(Ordering::Relaxed),
            prefetch_hits: c.prefetch_hits.load(Ordering::Relaxed),
            context_tokens: c.context_tokens.load(Ordering::Relaxed),
            backend_cache,
            ttft: self.inner.ttft.snapshot(),
            prefill: self.inner.prefill.snapshot(),
            decode: self.inner.decode.snapshot(),
        }
    }

    pub fn render_prometheus(&self) -> String {
        let s = self.snapshot();
        let mut out = String::new();
        metric(
            &mut out,
            "rustmoe_requests_total",
            "counter",
            "Requests started",
            s.requests,
        );
        metric(
            &mut out,
            "rustmoe_requests_active",
            "gauge",
            "Requests in flight",
            s.active_requests,
        );
        metric(
            &mut out,
            "rustmoe_errors_total",
            "counter",
            "Request errors",
            s.errors,
        );
        metric(
            &mut out,
            "rustmoe_prompt_tokens_total",
            "counter",
            "Prompt tokens processed",
            s.prompt_tokens,
        );
        metric(
            &mut out,
            "rustmoe_completion_tokens_total",
            "counter",
            "Completion tokens generated",
            s.completion_tokens,
        );
        metric(
            &mut out,
            "rustmoe_decode_steps_total",
            "counter",
            "Decode steps",
            s.decode_steps,
        );
        metric(
            &mut out,
            "rustmoe_cache_gpu_hits_total",
            "counter",
            "GPU expert cache hits",
            s.gpu_cache_hits,
        );
        metric(
            &mut out,
            "rustmoe_cache_host_hits_total",
            "counter",
            "Host expert cache hits",
            s.host_cache_hits,
        );
        metric(
            &mut out,
            "rustmoe_cache_disk_misses_total",
            "counter",
            "Cold expert reads",
            s.disk_misses,
        );
        metric(
            &mut out,
            "rustmoe_cache_hit_rate",
            "gauge",
            "Expert cache hit rate",
            s.cache_hit_rate,
        );
        metric(
            &mut out,
            "rustmoe_disk_bytes_read_total",
            "counter",
            "Expert bytes read from cold store",
            s.disk_bytes,
        );
        metric(
            &mut out,
            "rustmoe_h2d_bytes_total",
            "counter",
            "Expert bytes copied to GPU",
            s.h2d_bytes,
        );
        metric(
            &mut out,
            "rustmoe_cache_evictions_total",
            "counter",
            "Expert cache evictions",
            s.cache_evictions,
        );
        metric(
            &mut out,
            "rustmoe_context_tokens",
            "gauge",
            "Current context tokens",
            s.context_tokens,
        );
        histogram(&mut out, "rustmoe_ttft_ms", "Time to first token", &s.ttft);
        histogram(&mut out, "rustmoe_prefill_ms", "Prefill duration", &s.prefill);
        histogram(
            &mut out,
            "rustmoe_decode_ms",
            "Per-token decode duration",
            &s.decode,
        );
        if let Some(cache) = &s.backend_cache {
            metric(
                &mut out,
                "rustmoe_llama_moe_cache_hits_total",
                "counter",
                "llama.cpp MoE cache hits",
                cache.hits,
            );
            metric(
                &mut out,
                "rustmoe_llama_moe_cache_misses_total",
                "counter",
                "llama.cpp MoE cache misses",
                cache.misses,
            );
            metric(
                &mut out,
                "rustmoe_llama_moe_cache_hit_rate",
                "gauge",
                "llama.cpp MoE cache hit rate",
                cache.hit_rate,
            );
            metric(
                &mut out,
                "rustmoe_llama_moe_cache_uploaded_bytes_total",
                "counter",
                "llama.cpp MoE cache uploaded bytes",
                cache.uploaded_bytes,
            );
        }
        out
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BackendCacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub uploaded_bytes: u64,
    pub copied_hits: u64,
    pub copied_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricsSnapshot {
    pub requests: u64,
    pub active_requests: u64,
    pub errors: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub prefill_chunks: u64,
    pub decode_steps: u64,
    pub gpu_cache_hits: u64,
    pub host_cache_hits: u64,
    pub disk_misses: u64,
    pub cache_hit_rate: f64,
    pub disk_bytes: u64,
    pub h2d_bytes: u64,
    pub cache_evictions: u64,
    pub prefetch_issued: u64,
    pub prefetch_hits: u64,
    pub context_tokens: u64,
    pub backend_cache: Option<BackendCacheMetrics>,
    pub ttft: LatencySnapshot,
    pub prefill: LatencySnapshot,
    pub decode: LatencySnapshot,
}

fn metric(out: &mut String, name: &str, kind: &str, help: &str, value: impl std::fmt::Display) {
    out.push_str("# HELP ");
    out.push_str(name);
    out.push(' ');
    out.push_str(help);
    out.push('\n');
    out.push_str("# TYPE ");
    out.push_str(name);
    out.push(' ');
    out.push_str(kind);
    out.push('\n');
    out.push_str(name);
    out.push(' ');
    out.push_str(&value.to_string());
    out.push('\n');
}

fn histogram(out: &mut String, name: &str, help: &str, snap: &LatencySnapshot) {
    out.push_str("# HELP ");
    out.push_str(name);
    out.push(' ');
    out.push_str(help);
    out.push('\n');
    out.push_str("# TYPE ");
    out.push_str(name);
    out.push_str(" histogram\n");
    let mut cumulative = 0u64;
    for (edge, count) in &snap.buckets {
        cumulative += *count;
        out.push_str(name);
        out.push_str("_bucket{le=\"");
        out.push_str(&edge.to_string());
        out.push_str("\"} ");
        out.push_str(&cumulative.to_string());
        out.push('\n');
    }
    out.push_str(name);
    out.push_str("_bucket{le=\"+Inf\"} ");
    out.push_str(&snap.count.to_string());
    out.push('\n');
    out.push_str(name);
    out.push_str("_sum ");
    out.push_str(&snap.sum_ms.to_string());
    out.push('\n');
    out.push_str(name);
    out.push_str("_count ");
    out.push_str(&snap.count.to_string());
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_and_hit_rate() {
        let m = Metrics::new();
        m.request_started();
        m.add_prompt_tokens(7);
        m.add_completion_tokens(3);
        m.add_scheduler_step(4, 2, 2, 1024, 2048);
        m.request_finished();
        let s = m.snapshot();
        assert_eq!(s.requests, 1);
        assert_eq!(s.active_requests, 0);
        assert_eq!(s.prompt_tokens, 7);
        assert_eq!(s.cache_hit_rate, 0.75);
        assert_eq!(s.h2d_bytes, 2048);
    }

    #[test]
    fn backend_cache_is_exposed() {
        let m = Metrics::new();
        m.set_backend_cache(221, 1315, 9_924_618_240, 0, 0);
        let s = m.snapshot();
        let cache = s.backend_cache.expect("cache stats");
        assert_eq!(cache.hits, 221);
        assert_eq!(cache.misses, 1315);
        assert!((cache.hit_rate - 0.1439).abs() < 0.001);
        let text = m.render_prometheus();
        assert!(text.contains("rustmoe_llama_moe_cache_hits_total 221"));
        assert!(text.contains("rustmoe_llama_moe_cache_misses_total 1315"));
    }

    #[test]
    fn histogram_is_cumulative_in_prometheus() {
        let m = Metrics::new();
        m.observe_decode(Duration::from_millis(3));
        m.observe_decode(Duration::from_millis(30));
        let text = m.render_prometheus();
        assert!(text.contains("rustmoe_decode_ms_bucket{le=\"5\"} 1"));
        assert!(text.contains("rustmoe_decode_ms_bucket{le=\"50\"} 2"));
        assert!(text.contains("rustmoe_decode_ms_count 2"));
        let s = m.snapshot();
        assert_eq!(s.decode.count, 2);
        assert!(s.decode.mean_ms >= 16.0);
    }
}
