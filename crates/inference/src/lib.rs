//! Prefill, decode, sampling, and streamed session orchestration.

use rustmoe_backend::{BackendCacheStats, BackendError, BackendOutput, InferenceBackend, ModelState};
use rustmoe_model::ModelDescriptor;
use rustmoe_scheduler::{ExpertScheduler, SchedulerError, SchedulerStats, StepPlan};
use rustmoe_telemetry::{Metrics, MetricsSnapshot};
use rustmoe_tokenizer::{BpeTokenizer, TokenizerError};
use serde::Serialize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("backend error: {0}")]
    Backend(#[from] BackendError),
    #[error("scheduler error: {0}")]
    Scheduler(#[from] SchedulerError),
    #[error("prompt is empty after tokenization")]
    EmptyPrompt,
    #[error("max_tokens must be greater than zero")]
    NoTokens,
    #[error("context overflow: prompt {prompt} + max_tokens {max} > limit {limit}")]
    ContextLength { prompt: u64, max: u64, limit: u64 },
    #[error("invalid sampling configuration: {0}")]
    Sampling(String),
    #[error("scheduler does not match model: {0}")]
    SchedulerMismatch(String),
}

#[derive(Clone, Debug, Serialize)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub repetition_last_n: usize,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            stop_token_ids: Vec::new(),
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            repetition_last_n: 64,
            seed: 0xD5E1_4A11_1017_2026,
        }
    }
}

impl GenerationConfig {
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.max_tokens == 0 {
            return Err(InferenceError::NoTokens);
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(InferenceError::Sampling(
                "temperature must be finite and >= 0".into(),
            ));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(InferenceError::Sampling("top_p must be in [0, 1]".into()));
        }
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            return Err(InferenceError::Sampling("min_p must be in [0, 1]".into()));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(InferenceError::Sampling(
                "repetition_penalty must be finite and > 0".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

#[derive(Clone, Debug, Serialize)]
pub struct TokenEvent {
    pub index: usize,
    pub id: u32,
    pub text: String,
}

#[derive(Clone, Debug, Serialize)]
pub enum GenerationEvent {
    Token(TokenEvent),
    Text {
        text: String,
    },
    Finished {
        reason: FinishReason,
        completion_tokens: usize,
        elapsed_ms: f64,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct GenerationResult {
    pub prompt_tokens: Vec<u32>,
    pub tokens: Vec<u32>,
    pub text: String,
    pub finish_reason: FinishReason,
    pub prompt_token_count: u64,
    pub completion_token_count: u64,
    pub ttft_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub backend: String,
    pub simulation: bool,
    pub backend_cache: Option<BackendCacheStats>,
    pub scheduler: SchedulerStats,
    pub metrics: MetricsSnapshot,
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub context_limit: u64,
    pub prefill_chunk: usize,
}

impl EngineConfig {
    pub fn new(context_limit: u64, prefill_chunk: usize) -> Self {
        Self {
            context_limit,
            prefill_chunk: prefill_chunk.max(1),
        }
    }
}

pub struct Engine {
    model: Arc<ModelDescriptor>,
    tokenizer: Arc<BpeTokenizer>,
    backend: Box<dyn InferenceBackend>,
    scheduler: ExpertScheduler,
    metrics: Metrics,
    config: EngineConfig,
    state: ModelState,
}

impl Engine {
    pub fn new(
        model: Arc<ModelDescriptor>,
        tokenizer: Arc<BpeTokenizer>,
        mut backend: Box<dyn InferenceBackend>,
        scheduler: ExpertScheduler,
        metrics: Metrics,
        config: EngineConfig,
    ) -> Result<Self, InferenceError> {
        if scheduler.config().expected_layers != model.config.block_count {
            return Err(InferenceError::SchedulerMismatch(format!(
                "scheduler has {} layers, model has {}",
                scheduler.config().expected_layers,
                model.config.block_count
            )));
        }
        if scheduler.config().expected_top_k != model.config.experts.used_count as usize {
            return Err(InferenceError::SchedulerMismatch("Top-K differs".into()));
        }
        if config.context_limit == 0 || config.context_limit > model.config.context_length {
            return Err(InferenceError::ContextLength {
                prompt: 0,
                max: 0,
                limit: model.config.context_length,
            });
        }
        if tokenizer.vocab_size() as u64 != model.config.vocab_size {
            return Err(InferenceError::SchedulerMismatch(format!(
                "tokenizer vocab {} differs from model vocab {}",
                tokenizer.vocab_size(),
                model.config.vocab_size
            )));
        }
        backend.load_model(&model)?;
        let state = ModelState::new(config.context_limit);
        Ok(Self {
            model,
            tokenizer,
            backend,
            scheduler,
            metrics,
            config,
            state,
        })
    }

    pub fn model(&self) -> &ModelDescriptor {
        &self.model
    }

    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    pub fn is_simulation(&self) -> bool {
        self.backend.name() == "simulation"
    }

    pub fn scheduler(&self) -> &ExpertScheduler {
        &self.scheduler
    }

    pub fn scheduler_mut(&mut self) -> &mut ExpertScheduler {
        &mut self.scheduler
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    pub fn backend_cache_stats(&self) -> Option<BackendCacheStats> {
        self.backend.cache_stats()
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        cfg: &GenerationConfig,
    ) -> Result<GenerationResult, InferenceError> {
        self.generate_with(prompt, cfg, |_| true)
    }

    pub fn generate_with<F>(
        &mut self,
        prompt: &str,
        cfg: &GenerationConfig,
        mut on_event: F,
    ) -> Result<GenerationResult, InferenceError>
    where
        F: FnMut(GenerationEvent) -> bool,
    {
        self.metrics.request_started();
        let result = self.generate_inner(prompt, cfg, &mut on_event);
        if result.is_err() {
            self.metrics.error();
        }
        self.metrics.request_finished();
        result
    }

    fn generate_inner<F>(
        &mut self,
        prompt: &str,
        cfg: &GenerationConfig,
        on_event: &mut F,
    ) -> Result<GenerationResult, InferenceError>
    where
        F: FnMut(GenerationEvent) -> bool,
    {
        cfg.validate()?;
        let prompt_tokens = match self.backend.tokenize(prompt)? {
            Some(tokens) => tokens,
            None => self.tokenizer.encode(prompt),
        };
        if prompt_tokens.is_empty() {
            return Err(InferenceError::EmptyPrompt);
        }
        let prompt_len = prompt_tokens.len() as u64;
        let limit = self.config.context_limit;
        if prompt_len.saturating_add(cfg.max_tokens as u64) > limit {
            return Err(InferenceError::ContextLength {
                prompt: prompt_len,
                max: cfg.max_tokens as u64,
                limit,
            });
        }

        let total_start = Instant::now();
        self.state.reset();
        self.backend.reset();
        self.metrics.add_prompt_tokens(prompt_len);

        let prefill_start = Instant::now();
        let mut last: Option<BackendOutput> = None;
        for chunk in prompt_tokens.chunks(self.config.prefill_chunk) {
            let out = self.backend.prefill(chunk, &mut self.state)?;
            self.record_scheduler(&out)?;
            self.metrics.add_prefill_chunk();
            last = Some(out);
        }
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        self.metrics.observe_prefill(prefill_start.elapsed());
        let mut output = last.ok_or(InferenceError::EmptyPrompt)?;

        let mut sampler = Sampler::new(cfg.seed);
        let mut generated = Vec::new();
        let mut text = String::new();
        let mut pending_bytes = Vec::new();
        let mut finish = FinishReason::Length;
        let mut ttft = None;
        let mut decode_time = Duration::ZERO;
        let stop_ids: Vec<u32> = self
            .tokenizer
            .eos_token_id
            .into_iter()
            .chain(cfg.stop_token_ids.iter().copied())
            .collect();

        while generated.len() < cfg.max_tokens {
            let step_start = Instant::now();
            let token = sampler.sample(&output.logits, &self.state.token_history, cfg);
            if stop_ids.contains(&token) {
                finish = FinishReason::Stop;
                break;
            }
            if ttft.is_none() {
                let elapsed = total_start.elapsed();
                ttft = Some(elapsed);
                self.metrics.observe_ttft(elapsed);
            }
            generated.push(token);
            let decoded = self
                .backend
                .decode_token_bytes(token)
                .unwrap_or_else(|| self.tokenizer.decode_bytes(&[token]));
            pending_bytes.extend_from_slice(&decoded);
            let delta = drain_utf8(&mut pending_bytes);
            if !delta.is_empty() {
                text.push_str(&delta);
            }
            self.metrics.add_completion_tokens(1);
            self.metrics.set_context_tokens(self.state.position + 1);
            let keep_going = on_event(GenerationEvent::Token(TokenEvent {
                index: generated.len() - 1,
                id: token,
                text: delta,
            }));
            if !keep_going {
                finish = FinishReason::Cancelled;
                break;
            }
            if generated.len() >= cfg.max_tokens {
                finish = FinishReason::Length;
                break;
            }
            if self.state.position >= self.config.context_limit {
                finish = FinishReason::Length;
                break;
            }
            let out = self.backend.decode_step(token, &mut self.state)?;
            self.record_scheduler(&out)?;
            self.metrics.add_decode_step();
            let elapsed = step_start.elapsed();
            decode_time += elapsed;
            self.metrics.observe_decode(elapsed);
            output = out;
        }

        let tail = String::from_utf8_lossy(&pending_bytes).into_owned();
        if !tail.is_empty() {
            text.push_str(&tail);
            on_event(GenerationEvent::Text { text: tail });
        }
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = ttft.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(total_ms);
        let backend_cache = self.backend.cache_stats();
        if let Some(cache) = &backend_cache {
            self.metrics.set_backend_cache(
                cache.hits,
                cache.misses,
                cache.uploaded_bytes,
                cache.copied_hits,
                cache.copied_bytes,
            );
        }
        let result = GenerationResult {
            prompt_tokens,
            tokens: generated.clone(),
            text,
            finish_reason: finish,
            prompt_token_count: prompt_len,
            completion_token_count: generated.len() as u64,
            ttft_ms,
            prefill_ms,
            decode_ms: decode_time.as_secs_f64() * 1000.0,
            total_ms,
            backend: self.backend.name().to_string(),
            simulation: self.is_simulation(),
            backend_cache,
            scheduler: self.scheduler.stats().clone(),
            metrics: self.metrics.snapshot(),
        };
        on_event(GenerationEvent::Finished {
            reason: finish,
            completion_tokens: generated.len(),
            elapsed_ms: total_ms,
        });
        Ok(result)
    }

    fn record_scheduler(&mut self, output: &BackendOutput) -> Result<(), InferenceError> {
        if !self.backend.reports_routes() {
            return Ok(());
        }
        let plan: StepPlan = self.scheduler.plan_step(&output.routes, None)?;
        self.metrics.add_scheduler_step(
            plan.gpu_hits,
            plan.host_hits,
            plan.disk_misses,
            plan.disk_bytes,
            plan.h2d_bytes,
        );
        self.metrics
            .add_prefetch(plan.prefetch_issued, plan.prefetch_hits);
        self.metrics.add_evictions(plan.evictions);
        Ok(())
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub struct Sampler {
    rng: Rng,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self { rng: Rng::new(seed) }
    }

    pub fn sample(&mut self, logits: &[f32], history: &[u32], cfg: &GenerationConfig) -> u32 {
        let mut scores = logits.to_vec();
        apply_repetition_penalty(
            &mut scores,
            history,
            cfg.repetition_penalty,
            cfg.repetition_last_n,
        );
        if cfg.temperature <= 1e-6 {
            return argmax(&scores);
        }
        let inv_temp = 1.0 / cfg.temperature as f64;
        for s in &mut scores {
            *s = if s.is_finite() {
                (*s as f64 * inv_temp) as f32
            } else {
                -1e30
            };
        }
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        if cfg.top_k > 0 && cfg.top_k < scores.len() {
            let k = cfg.top_k;
            indices.select_nth_unstable_by(k - 1, |a, b| {
                scores[*b]
                    .partial_cmp(&scores[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            indices.truncate(k);
        }
        indices.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });
        let max = indices
            .iter()
            .map(|&i| scores[i] as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<(usize, f64)> = indices
            .iter()
            .map(|&i| (i, (scores[i] as f64 - max).exp()))
            .collect();
        let sum: f64 = probs.iter().map(|(_, p)| *p).sum();
        if !sum.is_finite() || sum <= 0.0 {
            return indices.first().copied().unwrap_or(0) as u32;
        }
        for (_, p) in &mut probs {
            *p /= sum;
        }
        if cfg.min_p > 0.0 {
            let floor = cfg.min_p as f64 * probs[0].1;
            probs.retain(|(_, p)| *p >= floor);
        }
        let mut kept = Vec::new();
        let mut cumulative = 0.0;
        for (idx, p) in probs {
            kept.push((idx, p));
            cumulative += p;
            if cumulative >= cfg.top_p as f64 {
                break;
            }
        }
        if kept.is_empty() {
            return 0;
        }
        let total: f64 = kept.iter().map(|(_, p)| *p).sum();
        let mut target = self.rng.next_f64() * total;
        for (idx, p) in &kept {
            if target <= *p {
                return *idx as u32;
            }
            target -= *p;
        }
        kept.last().map(|(i, _)| *i as u32).unwrap_or(0)
    }
}

fn apply_repetition_penalty(scores: &mut [f32], history: &[u32], penalty: f32, last_n: usize) {
    if (penalty - 1.0).abs() < f32::EPSILON || penalty <= 0.0 || last_n == 0 {
        return;
    }
    let start = history.len().saturating_sub(last_n);
    let mut seen = std::collections::BTreeSet::new();
    for &token in &history[start..] {
        if !seen.insert(token) {
            continue;
        }
        if let Some(score) = scores.get_mut(token as usize) {
            if *score > 0.0 {
                *score /= penalty;
            } else {
                *score *= penalty;
            }
        }
    }
}

fn argmax(scores: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, s) in scores.iter().enumerate() {
        if *s > best_score {
            best_score = *s;
            best = i;
        }
    }
    best as u32
}

/// Decode as much valid UTF-8 as possible, retaining an incomplete trailing
/// sequence for the next token.
fn drain_utf8(pending: &mut Vec<u8>) -> String {
    let mut out = String::new();
    loop {
        match std::str::from_utf8(pending) {
            Ok(s) => {
                out.push_str(s);
                pending.clear();
                return out;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid > 0 {
                    let s = std::str::from_utf8(&pending[..valid]).unwrap();
                    out.push_str(s);
                    pending.drain(..valid);
                }
                if let Some(len) = e.error_len() {
                    out.push('\u{FFFD}');
                    pending.drain(..len.min(pending.len()));
                } else {
                    return out;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampler_picks_argmax() {
        let mut s = Sampler::new(1);
        let cfg = GenerationConfig {
            temperature: 0.0,
            ..GenerationConfig::default()
        };
        assert_eq!(s.sample(&[0.1, 0.9, 0.2], &[], &cfg), 1);
    }

    #[test]
    fn sampling_is_deterministic_for_seed() {
        let cfg = GenerationConfig {
            temperature: 0.8,
            seed: 7,
            ..GenerationConfig::default()
        };
        let logits = vec![0.0, 0.2, 0.4, 0.1];
        let mut a = Sampler::new(cfg.seed);
        let mut b = Sampler::new(cfg.seed);
        for _ in 0..16 {
            assert_eq!(a.sample(&logits, &[], &cfg), b.sample(&logits, &[], &cfg));
        }
    }

    #[test]
    fn top_k_never_returns_outside_set() {
        let cfg = GenerationConfig {
            temperature: 1.0,
            top_k: 2,
            seed: 4,
            ..GenerationConfig::default()
        };
        let logits = vec![10.0, 0.0, 9.0, -10.0];
        let mut s = Sampler::new(4);
        for _ in 0..32 {
            let t = s.sample(&logits, &[], &cfg);
            assert!(t == 0 || t == 2);
        }
    }

    #[test]
    fn utf8_split_across_tokens_is_buffered() {
        let mut pending = Vec::new();
        // Euro sign split into its three bytes.
        pending.extend_from_slice(&[0xE2]);
        assert_eq!(drain_utf8(&mut pending), "");
        pending.extend_from_slice(&[0x82, 0xAC]);
        assert_eq!(drain_utf8(&mut pending), "\u{20AC}");
    }

    #[test]
    fn repetition_penalty_reduces_repeat() {
        let mut scores = vec![5.0, 4.0];
        apply_repetition_penalty(&mut scores, &[0], 2.0, 8);
        assert_eq!(scores[0], 2.5);
        assert_eq!(scores[1], 4.0);
    }
}
