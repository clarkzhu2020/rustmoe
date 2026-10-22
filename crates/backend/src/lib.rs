//! Inference backend boundary and deterministic simulation backend.
//!
//! The simulation backend exercises the runtime end to end (prefill, decode,
//! routing, sampling, SSE, metrics) but it is not a neural network. Its output
//! is deliberately labelled as simulation in every API response.

use rustmoe_model::ModelDescriptor;
use rustmoe_scheduler::{ExpertChoice, LayerRoute};
use serde::Serialize;
use std::any::Any;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend is not loaded")]
    NotLoaded,
    #[error("prefill requires at least one token")]
    EmptyPrefill,
    #[error("context overflow: position {position} + {tokens} > limit {limit}")]
    ContextOverflow { position: u64, tokens: u64, limit: u64 },
    #[error("backend failure: {0}")]
    Failed(String),
}

pub struct ModelState {
    pub context_limit: u64,
    pub position: u64,
    pub token_history: Vec<u32>,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub backend_state: Option<Box<dyn Any + Send>>,
}

impl ModelState {
    pub fn new(context_limit: u64) -> Self {
        Self {
            context_limit,
            position: 0,
            token_history: Vec::new(),
            prompt_tokens: 0,
            generated_tokens: 0,
            backend_state: None,
        }
    }

    pub fn remaining(&self) -> u64 {
        self.context_limit.saturating_sub(self.position)
    }

    pub fn reset(&mut self) {
        self.position = 0;
        self.token_history.clear();
        self.prompt_tokens = 0;
        self.generated_tokens = 0;
        self.backend_state = None;
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct BackendCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub uploaded_bytes: u64,
    pub copied_hits: u64,
    pub copied_bytes: u64,
}

impl BackendCacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BackendOutput {
    pub logits: Vec<f32>,
    pub routes: Vec<LayerRoute>,
}

pub trait InferenceBackend: Send {
    fn name(&self) -> &str;
    fn load_model(&mut self, model: &ModelDescriptor) -> Result<(), BackendError>;
    fn prefill(&mut self, tokens: &[u32], state: &mut ModelState) -> Result<BackendOutput, BackendError>;
    fn decode_step(&mut self, token: u32, state: &mut ModelState) -> Result<BackendOutput, BackendError>;

    /// Release per-session backend state before a new request. The default is
    /// correct for stateless backends.
    fn reset(&mut self) {}

    /// Exact tokenizer supplied by the backend, when it owns one. This avoids
    /// using an approximate pre-tokenizer for a reference implementation.
    fn tokenize(&self, _text: &str) -> Result<Option<Vec<u32>>, BackendError> {
        Ok(None)
    }

    /// Decode one token to raw bytes, when the backend owns the vocabulary.
    fn decode_token_bytes(&self, _token: u32) -> Option<Vec<u8>> {
        None
    }

    /// Whether the backend reports per-layer router selections to RustMoE.
    /// llama.cpp owns its router internally, so its backend returns false.
    fn reports_routes(&self) -> bool {
        true
    }

    /// Cumulative internal cache counters, when the backend owns a MoE cache.
    fn cache_stats(&self) -> Option<BackendCacheStats> {
        None
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SimulationSpec {
    pub block_count: usize,
    pub hidden_size: u64,
    pub vocab_size: u64,
    pub expert_count: u32,
    pub top_k: u32,
    pub shared_count: u32,
    pub routed_scale: f32,
    pub context_limit: u64,
    pub eos_token_id: Option<u32>,
}

impl SimulationSpec {
    pub fn from_model(model: &ModelDescriptor) -> Self {
        Self {
            block_count: model.config.block_count,
            hidden_size: model.config.hidden_size,
            vocab_size: model.config.vocab_size,
            expert_count: model.config.experts.count,
            top_k: model.config.experts.used_count,
            shared_count: model.config.experts.shared_count,
            routed_scale: model.config.experts.weights_scale,
            context_limit: model.config.context_length,
            eos_token_id: model.config.tokenizer.eos_token_id,
        }
    }
}

/// Deterministic, allocation-bounded mock of the decode loop. It does not
/// pretend to implement attention, MLA, Engram, or expert GEMM.
pub struct SimulatedBackend {
    spec: Option<SimulationSpec>,
    seed: u64,
    loaded: bool,
}

impl SimulatedBackend {
    pub fn new(seed: u64) -> Self {
        Self {
            spec: None,
            seed,
            loaded: false,
        }
    }

    pub fn from_spec(spec: SimulationSpec, seed: u64) -> Self {
        Self {
            spec: Some(spec),
            seed,
            loaded: false,
        }
    }

    pub fn spec(&self) -> Option<&SimulationSpec> {
        self.spec.as_ref()
    }

    fn spec_ref(&self) -> Result<&SimulationSpec, BackendError> {
        if !self.loaded {
            return Err(BackendError::NotLoaded);
        }
        self.spec.as_ref().ok_or(BackendError::NotLoaded)
    }

    fn step(
        &self,
        token: u32,
        state: &mut ModelState,
        advance: u64,
        output: BackendOutputFn,
    ) -> Result<BackendOutput, BackendError> {
        let spec = self.spec_ref()?;
        if state.position.saturating_add(advance) > state.context_limit {
            return Err(BackendError::ContextOverflow {
                position: state.position,
                tokens: advance,
                limit: state.context_limit,
            });
        }
        let position = state.position;
        let logits = self.logits(spec, token, position);
        let routes = self.routes(spec, token, position);
        state.position += advance;
        state.token_history.push(token);
        output(&state.token_history);
        Ok(BackendOutput { logits, routes })
    }

    fn hash(&self, token: u32, position: u64, layer: usize, expert: u32) -> u64 {
        let mut x = self.seed
            ^ (token as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ position.wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ (layer as u64).wrapping_mul(0xA24B_AED4_963E_E407)
            ^ (expert as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        x
    }

    fn unit(&self, token: u32, position: u64, layer: usize, expert: u32) -> f32 {
        (self.hash(token, position, layer, expert) >> 40) as f32 / (1u64 << 24) as f32
    }

    fn logits(&self, spec: &SimulationSpec, token: u32, position: u64) -> Vec<f32> {
        let mut logits = Vec::with_capacity(spec.vocab_size as usize);
        for id in 0..spec.vocab_size {
            let u = self.unit(token, position, 0xffff, id as u32);
            let bias = if id == token as u64 { 0.8 } else { 0.0 };
            logits.push((u - 0.5) * 8.0 + bias);
        }
        logits
    }

    fn routes(&self, spec: &SimulationSpec, token: u32, position: u64) -> Vec<LayerRoute> {
        let k = spec.top_k as usize;
        let mut out = Vec::with_capacity(spec.block_count);
        for layer in 0..spec.block_count {
            let mut scored: Vec<(f32, u32)> = (0..spec.expert_count)
                .map(|expert| {
                    let u = self.unit(token, position, layer, expert) as f64;
                    // sqrtsoftplus(x) is monotone; use a routed score proxy.
                    let score = ((1.0 + (-u).exp()).ln()).sqrt() as f32;
                    (score, expert)
                })
                .collect();
            scored.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            let chosen = &scored[..k.min(scored.len())];
            let sum: f32 = chosen.iter().map(|(s, _)| *s).sum::<f32>().max(1e-12);
            let experts = chosen
                .iter()
                .map(|(s, id)| ExpertChoice {
                    expert: *id,
                    weight: *s / sum * spec.routed_scale,
                })
                .collect();
            out.push(LayerRoute {
                layer: layer as u16,
                experts,
                shared_expert: spec.shared_count > 0,
            });
        }
        out
    }
}

impl InferenceBackend for SimulatedBackend {
    fn name(&self) -> &str {
        "simulation"
    }

    fn load_model(&mut self, model: &ModelDescriptor) -> Result<(), BackendError> {
        self.spec = Some(SimulationSpec::from_model(model));
        self.loaded = true;
        Ok(())
    }

    fn prefill(&mut self, tokens: &[u32], state: &mut ModelState) -> Result<BackendOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::EmptyPrefill);
        }
        let len = tokens.len() as u64;
        let last = *tokens.last().unwrap();
        let spec = self.spec_ref()?;
        if state.position.saturating_add(len) > state.context_limit {
            return Err(BackendError::ContextOverflow {
                position: state.position,
                tokens: len,
                limit: state.context_limit,
            });
        }
        let logits = self.logits(spec, last, state.position + len - 1);
        let routes = self.routes(spec, last, state.position + len - 1);
        state.position += len;
        state.prompt_tokens += len;
        state.token_history.extend_from_slice(tokens);
        Ok(BackendOutput { logits, routes })
    }

    fn decode_step(&mut self, token: u32, state: &mut ModelState) -> Result<BackendOutput, BackendError> {
        let out = self.step(token, state, 1, |_| {})?;
        state.generated_tokens += 1;
        Ok(out)
    }
}

type BackendOutputFn = fn(&[u32]);

#[allow(dead_code)]
fn _assert_object_safe(_: &dyn InferenceBackend) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> SimulationSpec {
        SimulationSpec {
            block_count: 4,
            hidden_size: 64,
            vocab_size: 128,
            expert_count: 16,
            top_k: 3,
            shared_count: 1,
            routed_scale: 1.5,
            context_limit: 16,
            eos_token_id: Some(1),
        }
    }

    fn loaded(spec: SimulationSpec) -> SimulatedBackend {
        let mut b = SimulatedBackend::from_spec(spec, 42);
        b.loaded = true;
        b
    }

    #[test]
    fn prefill_produces_full_logits_and_routes() {
        let mut b = loaded(spec());
        let mut state = ModelState::new(16);
        let out = b.prefill(&[5, 6, 7], &mut state).unwrap();
        assert_eq!(out.logits.len(), 128);
        assert_eq!(out.routes.len(), 4);
        assert_eq!(out.routes[0].experts.len(), 3);
        assert_eq!(state.position, 3);
        assert_eq!(state.prompt_tokens, 3);
    }

    #[test]
    fn decode_is_deterministic() {
        let mut a = loaded(spec());
        let mut b = loaded(spec());
        let mut sa = ModelState::new(16);
        let mut sb = ModelState::new(16);
        a.prefill(&[1, 2], &mut sa).unwrap();
        b.prefill(&[1, 2], &mut sb).unwrap();
        let oa = a.decode_step(9, &mut sa).unwrap();
        let ob = b.decode_step(9, &mut sb).unwrap();
        assert_eq!(oa.logits, ob.logits);
        assert_eq!(oa.routes[0].experts.len(), ob.routes[0].experts.len());
    }

    #[test]
    fn context_overflow_is_an_error() {
        let mut b = loaded(spec());
        let mut state = ModelState::new(2);
        b.prefill(&[1, 2], &mut state).unwrap();
        assert!(matches!(
            b.decode_step(3, &mut state).unwrap_err(),
            BackendError::ContextOverflow { .. }
        ));
    }

    #[test]
    fn unloaded_backend_refuses_work() {
        let mut b = SimulatedBackend::new(1);
        let mut state = ModelState::new(8);
        assert!(matches!(
            b.prefill(&[1], &mut state).unwrap_err(),
            BackendError::NotLoaded
        ));
    }
}
