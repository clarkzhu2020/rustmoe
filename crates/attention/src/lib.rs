//! DeepSeek-V4.1 attention-state geometry and KV-budget estimation.
//!
//! This is a planning model, not a tensor allocator. It deliberately reports a
//! conservative upper bound so the memory planner does not under-reserve VRAM.
//! A native backend must replace it with the exact cache layout of its kernels.

use rustmoe_model::DeepSeekV41Config;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheDtype {
    F16,
    BF16,
    Q8_0,
}

impl CacheDtype {
    pub fn bytes_per_element(self) -> u64 {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::Q8_0 => 1,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "half" => Some(Self::F16),
            "bf16" | "bfloat16" => Some(Self::BF16),
            "q8_0" | "q8" => Some(Self::Q8_0),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MlaGeometry {
    pub hidden_size: u64,
    pub head_count: u32,
    pub key_length: u64,
    pub value_length: u64,
    pub rope_dimension: u64,
    pub q_lora_rank: u64,
    pub output_groups: u32,
    pub output_lora_rank: u64,
    pub sliding_window: u64,
}

impl MlaGeometry {
    pub fn from_config(config: &DeepSeekV41Config) -> Self {
        Self {
            hidden_size: config.hidden_size,
            head_count: config.attention.head_count,
            key_length: config.attention.key_length as u64,
            value_length: config.attention.value_length as u64,
            rope_dimension: config.rope.dimension_count as u64,
            q_lora_rank: config.attention.q_lora_rank as u64,
            output_groups: config.attention.output_group_count,
            output_lora_rank: config.attention.output_lora_rank as u64,
            sliding_window: config.attention.sliding_window as u64,
        }
    }

    /// One token's raw MLA latent, key plus value, before compression.
    pub fn raw_token_elements(&self) -> u64 {
        self.key_length + self.value_length
    }

    pub fn compressed_token_elements(&self) -> u64 {
        self.key_length
    }

    pub fn index_token_elements(&self, index_key_length: u64) -> u64 {
        index_key_length
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct KvLayerPlan {
    pub layer: usize,
    pub compression_ratio: u32,
    pub raw_rows: u64,
    pub compressed_rows: u64,
    pub index_rows: u64,
    pub is_kv_source: bool,
    pub is_index_source: bool,
    pub has_engram: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct KvComponent {
    pub name: String,
    pub layer_count: u64,
    pub rows_per_layer: u64,
    pub row_bytes: u64,
    pub bytes: u64,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct KvPlan {
    pub context: u64,
    pub cache_dtype: CacheDtype,
    pub geometry: MlaGeometry,
    pub layers: Vec<KvLayerPlan>,
    pub components: Vec<KvComponent>,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub index_bytes: u64,
    pub auxiliary_bytes: u64,
    pub total_bytes: u64,
    pub bytes_per_token: f64,
    pub notes: Vec<String>,
}

impl KvPlan {
    pub fn estimate(config: &DeepSeekV41Config, context: u64, dtype: CacheDtype) -> Self {
        let geometry = MlaGeometry::from_config(config);
        let bpe = dtype.bytes_per_element();
        let context = context.max(1);
        let swa_rows = if geometry.sliding_window == 0 {
            context
        } else {
            context.min(geometry.sliding_window)
        };
        let mut layers = Vec::with_capacity(config.block_count);
        let mut compressed_groups: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut index_groups: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        for layer in 0..config.block_count {
            let ratio = config.compression_ratio(layer).max(1);
            let is_kv_source = config.is_kv_source(layer);
            let is_index_source = config.is_index_source(layer);
            let compressed_rows = if is_kv_source {
                context.div_ceil(ratio as u64)
            } else {
                0
            };
            let index_rows = if is_index_source {
                context.div_ceil(ratio as u64)
            } else {
                0
            };
            if is_kv_source {
                *compressed_groups.entry(ratio).or_insert(0) += 1;
            }
            if is_index_source {
                *index_groups.entry(ratio).or_insert(0) += 1;
            }
            layers.push(KvLayerPlan {
                layer,
                compression_ratio: config.compression_ratio(layer),
                raw_rows: swa_rows,
                compressed_rows,
                index_rows,
                is_kv_source,
                is_index_source,
                has_engram: config.is_engram_layer(layer),
            });
        }

        let mut components = Vec::new();
        let raw_row_bytes = geometry.raw_token_elements() * bpe;
        let raw_bytes = (config.block_count as u64)
            .saturating_mul(swa_rows)
            .saturating_mul(raw_row_bytes);
        components.push(KvComponent {
            name: "mla_raw_swa".into(),
            layer_count: config.block_count as u64,
            rows_per_layer: swa_rows,
            row_bytes: raw_row_bytes,
            bytes: raw_bytes,
            note: "sliding-window raw MLA latent; capped by attention.sliding_window".into(),
        });

        let compressed_row_bytes = geometry.compressed_token_elements() * bpe;
        let mut compressed_bytes = 0u64;
        for (ratio, count) in &compressed_groups {
            let rows = context.div_ceil(*ratio as u64);
            let bytes = count.saturating_mul(rows).saturating_mul(compressed_row_bytes);
            compressed_bytes = compressed_bytes.saturating_add(bytes);
            components.push(KvComponent {
                name: format!("compressed_kv_ratio_{ratio}"),
                layer_count: *count,
                rows_per_layer: rows,
                row_bytes: compressed_row_bytes,
                bytes,
                note: "one latent per compressed position; excludes backend workspace".into(),
            });
        }

        let index_row_bytes = config.attention.indexer.key_length as u64 * bpe;
        let mut index_bytes = 0u64;
        for (ratio, count) in &index_groups {
            let rows = context.div_ceil(*ratio as u64);
            let bytes = count.saturating_mul(rows).saturating_mul(index_row_bytes);
            index_bytes = index_bytes.saturating_add(bytes);
            components.push(KvComponent {
                name: format!("indexer_k_ratio_{ratio}"),
                layer_count: *count,
                rows_per_layer: rows,
                row_bytes: index_row_bytes,
                bytes,
                note: "indexer key cache; one row per compressed position".into(),
            });
        }

        // Engram history is one small integer per position. It is included so
        // the planner does not round it to zero at long context.
        let auxiliary_bytes = if config.engram.is_some() {
            context.saturating_mul(4)
        } else {
            0
        };
        if auxiliary_bytes > 0 {
            components.push(KvComponent {
                name: "engram_position_history".into(),
                layer_count: 1,
                rows_per_layer: context,
                row_bytes: 4,
                bytes: auxiliary_bytes,
                note: "lookback position history, not the Engram weight table".into(),
            });
        }

        let total_bytes = raw_bytes
            .saturating_add(compressed_bytes)
            .saturating_add(index_bytes)
            .saturating_add(auxiliary_bytes);
        let bytes_per_token = total_bytes as f64 / context as f64;
        let mut notes = vec![
            "KV numbers are a conservative planning upper bound derived from GGUF metadata.".into(),
            "The exact DSV4.1 cache layout (CSA/HCA/index streams and workspace) is backend-specific.".into(),
        ];
        if config.attention.sliding_window > 0 && context > geometry.sliding_window {
            notes.push(format!(
                "raw attention is capped at sliding_window={} tokens",
                geometry.sliding_window
            ));
        }
        Self {
            context,
            cache_dtype: dtype,
            geometry,
            layers,
            components,
            raw_bytes,
            compressed_bytes,
            index_bytes,
            auxiliary_bytes,
            total_bytes,
            bytes_per_token,
            notes,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "KV ctx={} dtype={:?} raw={} compressed={} index={} aux={} total={} ({:.1} B/token)",
            self.context,
            self.cache_dtype,
            crate::format_bytes(self.raw_bytes),
            crate::format_bytes(self.compressed_bytes),
            crate::format_bytes(self.index_bytes),
            crate::format_bytes(self.auxiliary_bytes),
            crate::format_bytes(self.total_bytes),
            self.bytes_per_token
        )
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AttentionStateDescriptor {
    pub position: u64,
    pub context_limit: u64,
    pub dtype: CacheDtype,
    pub plan: KvPlan,
}

impl AttentionStateDescriptor {
    pub fn new(config: &DeepSeekV41Config, context_limit: u64, dtype: CacheDtype) -> Self {
        Self {
            position: 0,
            context_limit,
            dtype,
            plan: KvPlan::estimate(config, context_limit, dtype),
        }
    }

    pub fn can_append(&self, tokens: u64) -> bool {
        self.position.saturating_add(tokens) <= self.context_limit
    }

    pub fn advance(&mut self, tokens: u64) -> bool {
        if !self.can_append(tokens) {
            return false;
        }
        self.position += tokens;
        true
    }

    pub fn remaining(&self) -> u64 {
        self.context_limit.saturating_sub(self.position)
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_token_uses_compression() {
        // A deliberately small geometry with two compressed layer groups.
        let config = test_config();
        let plan = KvPlan::estimate(&config, 8192, CacheDtype::F16);
        assert_eq!(plan.raw_bytes, 4 * 128 * (512 + 512) * 2);
        assert!(plan.compressed_bytes > 0);
        assert!(plan.total_bytes > plan.raw_bytes);
        assert!(plan.bytes_per_token > 0.0);
    }

    #[test]
    fn state_tracks_position() {
        let config = test_config();
        let mut state = AttentionStateDescriptor::new(&config, 8, CacheDtype::F16);
        assert!(state.advance(4));
        assert_eq!(state.remaining(), 4);
        assert!(!state.advance(5));
        assert_eq!(state.position, 4);
    }

    fn test_config() -> DeepSeekV41Config {
        use rustmoe_model::*;
        fn empty_tokenizer() -> TokenizerInfo {
            TokenizerInfo {
                model: None,
                pre_tokenizer: None,
                vocab_size: 4,
                bos_token_id: None,
                eos_token_id: None,
                padding_token_id: None,
                add_bos_token: None,
                add_eos_token: None,
                has_chat_template: false,
            }
        }
        DeepSeekV41Config {
            architecture: "deepseek41".into(),
            name: None,
            block_count: 4,
            context_length: 8192,
            hidden_size: 5120,
            vocab_size: 4,
            attention: AttentionConfig {
                head_count: 64,
                head_count_kv: 1,
                key_length: 512,
                value_length: 512,
                q_lora_rank: 1280,
                sliding_window: 128,
                output_group_count: 8,
                output_lora_rank: 1024,
                indexer: IndexerConfig {
                    head_count: 32,
                    key_length: 128,
                    top_k: 512,
                    source_layers: vec![1, 3],
                    candidate_source_layer: None,
                    candidate_top_k_blocks: None,
                    candidate_block_size: None,
                },
            },
            experts: ExpertConfig {
                count: 8,
                used_count: 2,
                shared_count: 1,
                intermediate_size: 64,
                gating_function: "sqrtsoftplus".into(),
                gating_function_id: 4,
                weights_scale: 1.5,
                weights_norm: true,
            },
            compress_ratios: vec![0, 2, 0, 1],
            kv_source_layers: vec![1, 3],
            index_source_layers: vec![1, 3],
            hyper_connection: HyperConnectionConfig {
                count: 4,
                sinkhorn_iterations: 20,
                epsilon: 1e-6,
            },
            engram: None,
            dspark: None,
            rms_norm_eps: 1e-20,
            rope: RopeConfig {
                freq_base: 10000.0,
                dimension_count: 64,
                scaling_type: None,
                scaling_factor: None,
                original_context_length: None,
                yarn_beta_fast: None,
                yarn_beta_slow: None,
                compress_freq_base: None,
            },
            swiglu_clamp_exp: vec![],
            swiglu_clamp_shexp: vec![],
            quantization: QuantizationInfo {
                file_type: None,
                quantization_version: None,
            },
            tokenizer: empty_tokenizer(),
        }
    }
}
