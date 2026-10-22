//! DeepSeek-V4.1-Flash model descriptor, metadata facts, tensor catalog
//! classification, and validation.
//!
//! The numeric constants and tensor shapes in this crate were checked against
//! the public UD-Q4_K_XL GGUF directory (model commit 60ad3651, October 2026).
//! They are still data-driven: every value comes from GGUF metadata, and a
//! mismatch is reported instead of silently coerced.

use rustmoe_gguf::{GgufError, GgufFile, GgufSet, Metadata, TensorInfo};
use rustmoe_quant::GgmlType;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const ARCH_DEEPSEEK41: &str = "deepseek41";
pub const ARCH_DSPARK: &str = "dflash";

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("GGUF error: {0}")]
    Gguf(#[from] GgufError),
    #[error("unsupported model architecture {0:?}; RustMoE implements deepseek41")]
    UnsupportedArchitecture(String),
    #[error("missing metadata key: {0}")]
    MissingKey(String),
    #[error("metadata key {key} has an unexpected value: {message}")]
    InvalidValue { key: String, message: String },
    #[error("model validation failed with {} error(s)", .0.len())]
    Validation(Vec<ValidationIssue>),
}

#[derive(Clone, Debug, Serialize)]
pub struct DeepSeekV41Config {
    pub architecture: String,
    pub name: Option<String>,
    pub block_count: usize,
    pub context_length: u64,
    pub hidden_size: u64,
    pub vocab_size: u64,
    pub attention: AttentionConfig,
    pub experts: ExpertConfig,
    pub compress_ratios: Vec<u32>,
    pub kv_source_layers: Vec<u32>,
    pub index_source_layers: Vec<u32>,
    pub hyper_connection: HyperConnectionConfig,
    pub engram: Option<EngramConfig>,
    pub dspark: Option<DsparkConfig>,
    pub rms_norm_eps: f64,
    pub rope: RopeConfig,
    pub swiglu_clamp_exp: Vec<f32>,
    pub swiglu_clamp_shexp: Vec<f32>,
    pub quantization: QuantizationInfo,
    pub tokenizer: TokenizerInfo,
}

#[derive(Clone, Debug, Serialize)]
pub struct AttentionConfig {
    pub head_count: u32,
    pub head_count_kv: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub q_lora_rank: u32,
    pub sliding_window: u32,
    pub output_group_count: u32,
    pub output_lora_rank: u32,
    pub indexer: IndexerConfig,
}

#[derive(Clone, Debug, Serialize)]
pub struct IndexerConfig {
    pub head_count: u32,
    pub key_length: u32,
    pub top_k: u32,
    pub source_layers: Vec<u32>,
    pub candidate_source_layer: Option<u32>,
    pub candidate_top_k_blocks: Option<u32>,
    pub candidate_block_size: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExpertConfig {
    pub count: u32,
    pub used_count: u32,
    pub shared_count: u32,
    pub intermediate_size: u32,
    pub gating_function: String,
    pub gating_function_id: u32,
    pub weights_scale: f32,
    pub weights_norm: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct HyperConnectionConfig {
    pub count: u32,
    pub sinkhorn_iterations: u32,
    pub epsilon: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct EngramConfig {
    pub layers: Vec<u32>,
    pub max_ngram_size: u32,
    pub heads_per_ngram: u32,
    pub head_dim: u32,
    pub table_rows: u64,
    pub token_vocab_size: u32,
    pub compressed_vocab_size: u32,
    pub pad_token_id: u32,
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
    pub layer_multipliers: Vec<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DsparkConfig {
    pub block_size: u32,
    pub noise_token_id: u32,
    pub target_layers: Vec<u32>,
    pub markov_rank: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct RopeConfig {
    pub freq_base: f32,
    pub dimension_count: u32,
    pub scaling_type: Option<String>,
    pub scaling_factor: Option<f32>,
    pub original_context_length: Option<u32>,
    pub yarn_beta_fast: Option<f32>,
    pub yarn_beta_slow: Option<f32>,
    pub compress_freq_base: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuantizationInfo {
    pub file_type: Option<u32>,
    pub quantization_version: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TokenizerInfo {
    pub model: Option<String>,
    pub pre_tokenizer: Option<String>,
    pub vocab_size: u64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub has_chat_template: bool,
}

impl DeepSeekV41Config {
    pub fn from_metadata(metadata: &Metadata) -> Result<Self, ModelError> {
        let architecture = metadata
            .get_str("general.architecture")
            .ok_or_else(|| ModelError::MissingKey("general.architecture".into()))?
            .to_string();
        if architecture != ARCH_DEEPSEEK41 {
            return Err(ModelError::UnsupportedArchitecture(architecture));
        }

        let block_count = require_u32(metadata, "deepseek41.block_count")? as usize;
        if block_count == 0 || block_count > 1024 {
            return Err(invalid("deepseek41.block_count", "must be in 1..=1024"));
        }
        let hidden_size = require_u32(metadata, "deepseek41.embedding_length")? as u64;
        let context_length = require_u32(metadata, "deepseek41.context_length")? as u64;
        let head_count = require_u32(metadata, "deepseek41.attention.head_count")?;
        let key_length = require_u32(metadata, "deepseek41.attention.key_length")?;
        let value_length = require_u32(metadata, "deepseek41.attention.value_length")?;
        let q_lora_rank = require_u32(metadata, "deepseek41.attention.q_lora_rank")?;
        let output_group_count = require_u32(metadata, "deepseek41.attention.output_group_count")?;
        let output_lora_rank = require_u32(metadata, "deepseek41.attention.output_lora_rank")?;
        let n_expert = require_u32(metadata, "deepseek41.expert_count")?;
        let n_used = require_u32(metadata, "deepseek41.expert_used_count")?;
        let n_shared = require_u32(metadata, "deepseek41.expert_shared_count")?;
        let expert_ffn = require_u32(metadata, "deepseek41.expert_feed_forward_length")?;
        if n_used == 0 || n_used > n_expert {
            return Err(invalid(
                "deepseek41.expert_used_count",
                "must be in 1..=expert_count",
            ));
        }
        if head_count == 0 || key_length == 0 || q_lora_rank == 0 {
            return Err(invalid(
                "deepseek41.attention.*",
                "head_count, key_length, and q_lora_rank must be non-zero",
            ));
        }
        if output_group_count == 0 || (head_count as u64 * key_length as u64) % output_group_count as u64 != 0
        {
            return Err(invalid(
                "deepseek41.attention.output_group_count",
                "must divide head_count * key_length",
            ));
        }

        let mut compress_ratios = metadata
            .get_u32_array("deepseek41.attention.compress_ratios")
            .ok_or_else(|| ModelError::MissingKey("deepseek41.attention.compress_ratios".into()))?;
        if compress_ratios.len() < block_count {
            return Err(invalid(
                "deepseek41.attention.compress_ratios",
                "length is shorter than block_count",
            ));
        }
        compress_ratios.truncate(block_count);

        let kv_source_layers = metadata
            .get_u32_array("deepseek41.attention.kv_source_layers")
            .unwrap_or_default();
        let index_source_layers = metadata
            .get_u32_array("deepseek41.attention.indexer.source_layers")
            .unwrap_or_default();
        validate_layer_list(
            "deepseek41.attention.kv_source_layers",
            &kv_source_layers,
            block_count,
        )?;
        validate_layer_list(
            "deepseek41.attention.indexer.source_layers",
            &index_source_layers,
            block_count,
        )?;

        let indexer = IndexerConfig {
            head_count: require_u32(metadata, "deepseek41.attention.indexer.head_count")?,
            key_length: require_u32(metadata, "deepseek41.attention.indexer.key_length")?,
            top_k: require_u32(metadata, "deepseek41.attention.indexer.top_k")?,
            source_layers: index_source_layers.clone(),
            candidate_source_layer: metadata.get_u32("deepseek41.attention.indexer.candidate_source_layer"),
            candidate_top_k_blocks: metadata.get_u32("deepseek41.attention.indexer.candidate_top_k_blocks"),
            candidate_block_size: metadata.get_u32("deepseek41.attention.indexer.candidate_block_size"),
        };

        let hyper_connection = HyperConnectionConfig {
            count: require_u32(metadata, "deepseek41.hyper_connection.count")?,
            sinkhorn_iterations: metadata
                .get_u32("deepseek41.hyper_connection.sinkhorn_iterations")
                .unwrap_or(20),
            epsilon: metadata
                .get_f32("deepseek41.hyper_connection.epsilon")
                .unwrap_or(1e-6),
        };
        if hyper_connection.count == 0 {
            return Err(invalid("deepseek41.hyper_connection.count", "must be non-zero"));
        }

        let tokenizer_tokens = metadata
            .get_array("tokenizer.ggml.tokens")
            .ok_or_else(|| ModelError::MissingKey("tokenizer.ggml.tokens".into()))?;
        let vocab_size = tokenizer_tokens.len() as u64;
        let tokenizer = TokenizerInfo {
            model: metadata.get_str("tokenizer.ggml.model").map(str::to_string),
            pre_tokenizer: metadata.get_str("tokenizer.ggml.pre").map(str::to_string),
            vocab_size,
            bos_token_id: metadata.get_u32("tokenizer.ggml.bos_token_id"),
            eos_token_id: metadata.get_u32("tokenizer.ggml.eos_token_id"),
            padding_token_id: metadata.get_u32("tokenizer.ggml.padding_token_id"),
            add_bos_token: metadata.get_bool("tokenizer.ggml.add_bos_token"),
            add_eos_token: metadata.get_bool("tokenizer.ggml.add_eos_token"),
            has_chat_template: metadata.contains_key("tokenizer.chat_template"),
        };

        let engram = parse_engram(metadata, block_count, hidden_size, hyper_connection.count)?;
        let dspark = parse_dspark(metadata, block_count);

        let gating_function_id = require_u32(metadata, "deepseek41.expert_gating_func")?;
        let gating_function = match gating_function_id {
            4 => "sqrtsoftplus",
            0 => "softmax",
            1 => "sigmoid",
            2 => "silu",
            other => {
                return Err(invalid(
                    "deepseek41.expert_gating_func",
                    &format!("unknown id {other}"),
                ))
            }
        }
        .to_string();

        let rope = RopeConfig {
            freq_base: metadata.get_f32("deepseek41.rope.freq_base").unwrap_or(10_000.0),
            dimension_count: require_u32(metadata, "deepseek41.rope.dimension_count")?,
            scaling_type: metadata
                .get_str("deepseek41.rope.scaling.type")
                .map(str::to_string),
            scaling_factor: metadata.get_f32("deepseek41.rope.scaling.factor"),
            original_context_length: metadata.get_u32("deepseek41.rope.scaling.original_context_length"),
            yarn_beta_fast: metadata.get_f32("deepseek41.rope.scaling.yarn_beta_fast"),
            yarn_beta_slow: metadata.get_f32("deepseek41.rope.scaling.yarn_beta_slow"),
            compress_freq_base: metadata.get_f32("deepseek41.attention.compress_rope_freq_base"),
        };

        let cfg = Self {
            architecture,
            name: metadata.get_str("general.name").map(str::to_string),
            block_count,
            context_length,
            hidden_size,
            vocab_size,
            attention: AttentionConfig {
                head_count,
                head_count_kv: metadata
                    .get_u32("deepseek41.attention.head_count_kv")
                    .unwrap_or(1),
                key_length,
                value_length,
                q_lora_rank,
                sliding_window: metadata
                    .get_u32("deepseek41.attention.sliding_window")
                    .unwrap_or(0),
                output_group_count,
                output_lora_rank,
                indexer,
            },
            experts: ExpertConfig {
                count: n_expert,
                used_count: n_used,
                shared_count: n_shared,
                intermediate_size: expert_ffn,
                gating_function,
                gating_function_id,
                weights_scale: metadata.get_f32("deepseek41.expert_weights_scale").unwrap_or(1.0),
                weights_norm: metadata
                    .get_bool("deepseek41.expert_weights_norm")
                    .unwrap_or(false),
            },
            compress_ratios,
            kv_source_layers,
            index_source_layers: index_source_layers.clone(),
            hyper_connection,
            engram,
            dspark,
            rms_norm_eps: metadata
                .get_f32("deepseek41.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-5) as f64,
            rope,
            swiglu_clamp_exp: metadata
                .get_f32_array("deepseek41.swiglu_clamp_exp")
                .unwrap_or_default(),
            swiglu_clamp_shexp: metadata
                .get_f32_array("deepseek41.swiglu_clamp_shexp")
                .unwrap_or_default(),
            quantization: QuantizationInfo {
                file_type: metadata.get_u32("general.file_type"),
                quantization_version: metadata.get_u32("general.quantization_version"),
            },
            tokenizer,
        };
        cfg.validate_facts()?;
        Ok(cfg)
    }

    pub fn validate_facts(&self) -> Result<(), ModelError> {
        if self.context_length < self.attention.sliding_window as u64 {
            return Err(invalid(
                "deepseek41.context_length",
                "must be at least attention.sliding_window",
            ));
        }
        if self.tokenizer.vocab_size == 0 {
            return Err(invalid("tokenizer.ggml.tokens", "empty vocabulary"));
        }
        if self.engram.as_ref().is_some_and(|e| e.table_rows == 0) {
            return Err(invalid("deepseek41.ple.head_vocab_sizes", "table is empty"));
        }
        Ok(())
    }

    pub fn is_kv_source(&self, layer: usize) -> bool {
        self.kv_source_layers.iter().any(|&l| l as usize == layer)
    }

    pub fn is_index_source(&self, layer: usize) -> bool {
        self.index_source_layers.iter().any(|&l| l as usize == layer)
    }

    pub fn compression_ratio(&self, layer: usize) -> u32 {
        self.compress_ratios.get(layer).copied().unwrap_or(0)
    }

    pub fn is_engram_layer(&self, layer: usize) -> bool {
        self.engram
            .as_ref()
            .is_some_and(|e| e.layers.iter().any(|&l| l as usize == layer))
    }

    /// Router experts selected by a token. This is the model contract; the
    /// scheduler must never reduce it to save bandwidth.
    pub fn experts_per_token(&self) -> u32 {
        self.experts.used_count
    }

    pub fn hidden_out(&self) -> u64 {
        self.hidden_size * self.hyper_connection.count as u64
    }

    pub fn q_head_width(&self) -> u64 {
        self.attention.head_count as u64 * self.attention.key_length as u64
    }

    pub fn o_width(&self) -> u64 {
        self.attention.output_group_count as u64 * self.attention.output_lora_rank as u64
    }
}

fn parse_engram(
    metadata: &Metadata,
    block_count: usize,
    hidden_size: u64,
    hc_mult: u32,
) -> Result<Option<EngramConfig>, ModelError> {
    let Some(layers) = metadata.get_u32_array("deepseek41.ple.layers") else {
        return Ok(None);
    };
    validate_layer_list("deepseek41.ple.layers", &layers, block_count)?;
    let head_vocab_sizes = metadata
        .get_u64_array("deepseek41.ple.head_vocab_sizes")
        .ok_or_else(|| ModelError::MissingKey("deepseek41.ple.head_vocab_sizes".into()))?;
    let head_offsets = metadata
        .get_u64_array("deepseek41.ple.head_offsets")
        .unwrap_or_default();
    let table_rows = head_vocab_sizes
        .iter()
        .try_fold(0u64, |a, &b| a.checked_add(b))
        .ok_or_else(|| invalid("deepseek41.ple.head_vocab_sizes", "row count overflow"))?;
    let head_dim = require_u32(metadata, "deepseek41.embedding_length_per_layer_input")?;
    let max_ngram_size = require_u32(metadata, "deepseek41.ple.ngram_size")?;
    let heads_per_ngram = require_u32(metadata, "deepseek41.ple.heads_per_ngram")?;
    if max_ngram_size < 2 || heads_per_ngram == 0 || head_dim == 0 {
        return Err(invalid(
            "deepseek41.ple.*",
            "ngram_size >= 2, heads_per_ngram > 0, head_dim > 0 are required",
        ));
    }
    let _ = (hidden_size, hc_mult);
    Ok(Some(EngramConfig {
        layers,
        max_ngram_size,
        heads_per_ngram,
        head_dim,
        table_rows,
        token_vocab_size: metadata.get_u32("deepseek41.engram.vocab_size").unwrap_or(0),
        compressed_vocab_size: metadata
            .get_u32("deepseek41.engram.compressed_vocab_size")
            .unwrap_or(0),
        pad_token_id: metadata.get_u32("deepseek41.engram.pad_token_id").unwrap_or(0),
        head_offsets,
        head_vocab_sizes,
        layer_multipliers: metadata
            .get_u64_array("deepseek41.ple.layer_multipliers")
            .unwrap_or_default(),
    }))
}

fn parse_dspark(metadata: &Metadata, block_count: usize) -> Option<DsparkConfig> {
    if !metadata.contains_key("deepseek41.dspark.block_size") {
        return None;
    }
    let target_layers = metadata
        .get_u32_array("deepseek41.dspark.target_layers")
        .unwrap_or_default();
    if target_layers.iter().any(|&l| l as usize >= block_count) {
        return None;
    }
    Some(DsparkConfig {
        block_size: metadata.get_u32("deepseek41.dspark.block_size").unwrap_or(5),
        noise_token_id: metadata.get_u32("deepseek41.dspark.noise_token_id").unwrap_or(0),
        target_layers,
        markov_rank: metadata.get_u32("deepseek41.dspark.markov_rank").unwrap_or(0),
        expert_count: metadata.get_u32("deepseek41.dspark.expert_count").unwrap_or(0),
        expert_used_count: metadata
            .get_u32("deepseek41.dspark.expert_used_count")
            .unwrap_or(0),
    })
}

fn require_u32(metadata: &Metadata, key: &str) -> Result<u32, ModelError> {
    metadata
        .get_u32(key)
        .ok_or_else(|| ModelError::MissingKey(key.to_string()))
}

fn invalid(key: &str, message: &str) -> ModelError {
    ModelError::InvalidValue {
        key: key.to_string(),
        message: message.to_string(),
    }
}

fn validate_layer_list(key: &str, layers: &[u32], block_count: usize) -> Result<(), ModelError> {
    let mut seen = BTreeSet::new();
    for &l in layers {
        if l as usize >= block_count {
            return Err(invalid(key, "contains a layer id >= block_count"));
        }
        if !seen.insert(l) {
            return Err(invalid(key, "contains duplicate layer ids"));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    TokenEmbedding,
    Output,
    Norm,
    Attention,
    Compressor,
    Indexer,
    Router,
    RoutedExpert,
    SharedExpert,
    HyperConnection,
    Engram,
    PleTable,
    Auxiliary,
    Unknown,
}

impl TensorRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TokenEmbedding => "token_embedding",
            Self::Output => "output",
            Self::Norm => "norm",
            Self::Attention => "attention",
            Self::Compressor => "compressor",
            Self::Indexer => "indexer",
            Self::Router => "router",
            Self::RoutedExpert => "routed_expert",
            Self::SharedExpert => "shared_expert",
            Self::HyperConnection => "hyper_connection",
            Self::Engram => "engram",
            Self::PleTable => "ple_table",
            Self::Auxiliary => "auxiliary",
            Self::Unknown => "unknown",
        }
    }

    pub fn is_expert(self) -> bool {
        matches!(self, Self::RoutedExpert | Self::SharedExpert)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorDescriptor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub nbytes: Option<u64>,
    pub elements: u64,
    pub shard: usize,
    pub offset: u64,
    pub data_offset: u64,
    pub role: TensorRole,
    pub layer: Option<usize>,
    pub expert_count: Option<u64>,
    pub bytes_per_expert: Option<u64>,
}

impl TensorDescriptor {
    pub fn from_info(info: &TensorInfo) -> Self {
        let role = classify_tensor(&info.name);
        let layer = layer_index(&info.name);
        let expert_count = if role == TensorRole::RoutedExpert {
            info.dims.last().copied()
        } else {
            None
        };
        let bytes_per_expert = match (expert_count, info.nbytes) {
            (Some(n), Some(bytes)) if n > 0 => Some(bytes / n),
            _ => None,
        };
        Self {
            name: info.name.clone(),
            dims: info.dims.clone(),
            ggml_type: info.ggml_type,
            nbytes: info.nbytes,
            elements: info.elements,
            shard: info.shard,
            offset: info.offset,
            data_offset: info.data_offset,
            role,
            layer,
            expert_count,
            bytes_per_expert,
        }
    }

    pub fn absolute_offset(&self) -> u64 {
        self.data_offset + self.offset
    }
}

pub fn classify_tensor(name: &str) -> TensorRole {
    if name == "per_layer_token_embd.weight" {
        return TensorRole::PleTable;
    }
    if name == "token_embd.weight" {
        return TensorRole::TokenEmbedding;
    }
    if name == "output.weight" {
        return TensorRole::Output;
    }
    if name.ends_with("_norm.weight") || name == "output_norm.weight" {
        return TensorRole::Norm;
    }
    if name.contains("engram") {
        return TensorRole::Engram;
    }
    if name.contains("indexer") {
        return TensorRole::Indexer;
    }
    if name.contains("attn_compressor") {
        return TensorRole::Compressor;
    }
    if name.contains(".hc_") || name.starts_with("hc_") {
        return TensorRole::HyperConnection;
    }
    if name.contains("ffn_gate_inp") || name.contains("exp_probs_b") || name.contains("ffn_gate_tid2eid") {
        return TensorRole::Router;
    }
    if name.contains("ffn_") && name.contains("_exps.weight") {
        return TensorRole::RoutedExpert;
    }
    if name.contains("ffn_") && name.contains("_shexp.weight") {
        return TensorRole::SharedExpert;
    }
    if name.contains("attn_") {
        return TensorRole::Attention;
    }
    if name.starts_with("nextn.") || name.contains(".nextn.") {
        return TensorRole::Auxiliary;
    }
    TensorRole::Unknown
}

pub fn layer_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let (digits, _) = rest.split_once('.')?;
    digits.parse().ok()
}

#[derive(Clone, Debug, Serialize)]
pub struct LayerSummary {
    pub layer: usize,
    pub compression_ratio: u32,
    pub is_kv_source: bool,
    pub is_index_source: bool,
    pub has_engram: bool,
    pub spine_bytes: u64,
    pub routed_expert_bytes: u64,
    pub shared_expert_bytes: u64,
    pub expert_tensor_bytes: u64,
    pub expert_count: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelSummary {
    pub tensor_count: usize,
    pub total_tensor_bytes: u64,
    pub routed_expert_bytes: u64,
    pub shared_expert_bytes: u64,
    pub dense_spine_bytes: u64,
    pub ple_table_bytes: u64,
    pub auxiliary_bytes: u64,
    pub expert_count_per_layer: u64,
    pub avg_expert_bytes: u64,
    pub min_expert_bytes: u64,
    pub max_expert_bytes: u64,
    pub layers: Vec<LayerSummary>,
    pub roles: BTreeMap<String, u64>,
    pub quant_types: BTreeMap<String, TypeBytes>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TypeBytes {
    pub tensors: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelDescriptor {
    pub path: PathBuf,
    pub config: DeepSeekV41Config,
    pub tensors: Vec<TensorDescriptor>,
    pub summary: ModelSummary,
    #[serde(skip)]
    pub gguf: GgufSet,
}

impl ModelDescriptor {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let path = path.as_ref().to_path_buf();
        let gguf = GgufSet::open(&path)?;
        Self::from_gguf(path, gguf)
    }

    pub fn from_gguf(path: PathBuf, gguf: GgufSet) -> Result<Self, ModelError> {
        let config = DeepSeekV41Config::from_metadata(&gguf.metadata)?;
        let tensors: Vec<TensorDescriptor> = gguf.tensors.iter().map(TensorDescriptor::from_info).collect();
        let summary = summarize(&config, &tensors);
        Ok(Self {
            path,
            config,
            tensors,
            summary,
            gguf,
        })
    }

    /// Load exactly one shard, without checking that the rest of a split set is
    /// present. Intended for metadata inspection and simulation only.
    pub fn load_single_shard(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let path = path.as_ref().to_path_buf();
        let file = GgufFile::parse_path(&path, &Default::default())?;
        let metadata = file.metadata.clone();
        let tensors = file.tensors.clone();
        let total_tensor_bytes = tensors.iter().map(|t| t.nbytes.unwrap_or(0)).sum();
        let total_file_bytes = file.file_len;
        let gguf = GgufSet {
            files: vec![file],
            primary: 0,
            metadata,
            tensors,
            total_file_bytes,
            total_tensor_bytes,
        };
        Self::from_gguf(path, gguf)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .ok()
            .map(|i| &self.tensors[i])
    }

    pub fn tensors_for_layer(&self, layer: usize) -> impl Iterator<Item = &TensorDescriptor> {
        self.tensors.iter().filter(move |t| t.layer == Some(layer))
    }

    pub fn expert_layout(&self, layer: usize) -> Option<ExpertLayout> {
        let mut gate = None;
        let mut down = None;
        let mut up = None;
        for t in self.tensors_for_layer(layer) {
            if t.role != TensorRole::RoutedExpert {
                continue;
            }
            if t.name.ends_with("ffn_gate_exps.weight") {
                gate = Some(t);
            } else if t.name.ends_with("ffn_down_exps.weight") {
                down = Some(t);
            } else if t.name.ends_with("ffn_up_exps.weight") {
                up = Some(t);
            }
        }
        let (gate, down, up) = (gate?, down?, up?);
        let count = gate.expert_count?;
        if down.expert_count != Some(count) || up.expert_count != Some(count) {
            return None;
        }
        let per = |t: &TensorDescriptor| t.bytes_per_expert.unwrap_or(0);
        Some(ExpertLayout {
            layer,
            expert_count: count,
            gate_bytes: per(gate),
            down_bytes: per(down),
            up_bytes: per(up),
            expert_bytes: per(gate) + per(down) + per(up),
            gate_offset: gate.absolute_offset(),
            down_offset: down.absolute_offset(),
            up_offset: up.absolute_offset(),
            gate_shard: gate.shard,
            down_shard: down.shard,
            up_shard: up.shard,
        })
    }

    pub fn validate(&self) -> ValidationReport {
        self.validate_with(&ValidationOptions::default())
    }

    pub fn validate_with(&self, options: &ValidationOptions) -> ValidationReport {
        let mut report = ValidationReport::new();
        self.validate_catalog(&mut report, options);
        report
    }

    fn validate_catalog(&self, report: &mut ValidationReport, options: &ValidationOptions) {
        if self.tensors.is_empty() {
            report.error(
                "catalog.empty",
                "tensor directory",
                "no tensors present; this looks like a metadata-only shard",
                "point RustMoE at shard 1 of a complete split set",
            );
            return;
        }
        let cfg = &self.config;
        let map: BTreeMap<&str, &TensorDescriptor> =
            self.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        let mut expected: BTreeSet<String> = BTreeSet::new();

        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        expect(
            report,
            &map,
            &mut expected,
            "token_embd.weight",
            &[hidden, vocab],
            TensorRole::TokenEmbedding,
            false,
            options,
        );
        expect(
            report,
            &map,
            &mut expected,
            "output.weight",
            &[hidden, vocab],
            TensorRole::Output,
            false,
            options,
        );
        expect(
            report,
            &map,
            &mut expected,
            "output_norm.weight",
            &[hidden],
            TensorRole::Norm,
            false,
            options,
        );

        if let Some(engram) = &cfg.engram {
            expect(
                report,
                &map,
                &mut expected,
                "per_layer_token_embd.weight",
                &[engram.head_dim as u64, engram.table_rows],
                TensorRole::PleTable,
                false,
                options,
            );
        }

        let q_lora = cfg.attention.q_lora_rank as u64;
        let n_head = cfg.attention.head_count as u64;
        let kv_len = cfg.attention.key_length as u64;
        let o_groups = cfg.attention.output_group_count as u64;
        let o_rank = cfg.attention.output_lora_rank as u64;
        let hc = cfg.hyper_connection.count as u64;
        let hc_dim = hidden * hc;
        let hc_mix = (2 + hc) * hc;
        let n_expert = cfg.experts.count as u64;
        let ffn = cfg.experts.intermediate_size as u64;
        let shared = cfg.experts.shared_count as u64;

        for layer in 0..cfg.block_count {
            let p = |s: &str| format!("blk.{layer}.{s}");
            macro_rules! req {
                ($name:expr, $dims:expr, $role:expr, $opt:expr $(,)?) => {
                    expect(
                        report,
                        &map,
                        &mut expected,
                        &$name,
                        &$dims,
                        $role,
                        $opt,
                        options,
                    )
                };
            }
            req!(p("attn_norm.weight"), [hidden], TensorRole::Norm, false,);
            req!(p("attn_sinks.weight"), [n_head], TensorRole::Attention, false,);
            req!(
                p("attn_q_a.weight"),
                [hidden, q_lora],
                TensorRole::Attention,
                false,
            );
            req!(p("attn_q_a_norm.weight"), [q_lora], TensorRole::Norm, false,);
            req!(
                p("attn_q_b.weight"),
                [q_lora, n_head * kv_len],
                TensorRole::Attention,
                false,
            );
            req!(
                p("attn_kv.weight"),
                [hidden, kv_len],
                TensorRole::Attention,
                false,
            );
            req!(p("attn_kv_a_norm.weight"), [kv_len], TensorRole::Norm, false,);
            req!(
                p("attn_output_a.weight"),
                [n_head * kv_len / o_groups, o_rank * o_groups],
                TensorRole::Attention,
                false,
            );
            req!(
                p("attn_output_b.weight"),
                [o_rank * o_groups, hidden],
                TensorRole::Attention,
                false,
            );
            for part in ["attn", "ffn"] {
                req!(
                    p(&format!("hc_{part}_fn.weight")),
                    [hc_dim, hc_mix],
                    TensorRole::HyperConnection,
                    false,
                );
                req!(
                    p(&format!("hc_{part}_base.weight")),
                    [hc_mix],
                    TensorRole::HyperConnection,
                    false,
                );
                req!(
                    p(&format!("hc_{part}_scale.weight")),
                    [3],
                    TensorRole::HyperConnection,
                    false,
                );
            }
            req!(p("ffn_norm.weight"), [hidden], TensorRole::Norm, false,);
            req!(
                p("ffn_gate_inp.weight"),
                [hidden, n_expert],
                TensorRole::Router,
                false,
            );
            req!(p("exp_probs_b.bias"), [n_expert], TensorRole::Router, false,);
            req!(p("exp_probs_b_vl.bias"), [n_expert], TensorRole::Router, true,);
            req!(
                p("ffn_gate_exps.weight"),
                [hidden, ffn, n_expert],
                TensorRole::RoutedExpert,
                false,
            );
            req!(
                p("ffn_down_exps.weight"),
                [ffn, hidden, n_expert],
                TensorRole::RoutedExpert,
                false,
            );
            req!(
                p("ffn_up_exps.weight"),
                [hidden, ffn, n_expert],
                TensorRole::RoutedExpert,
                false,
            );
            req!(
                p("ffn_gate_shexp.weight"),
                [hidden, ffn * shared],
                TensorRole::SharedExpert,
                false,
            );
            req!(
                p("ffn_down_shexp.weight"),
                [ffn * shared, hidden],
                TensorRole::SharedExpert,
                false,
            );
            req!(
                p("ffn_up_shexp.weight"),
                [hidden, ffn * shared],
                TensorRole::SharedExpert,
                false,
            );

            // compress_ratios may be non-zero on layers that are not KV
            // sources. Only source layers own compressor tensors. This was
            // checked against the real 40-layer catalog.
            let ratio = cfg.compression_ratio(layer);
            if cfg.is_kv_source(layer) {
                let coff = if ratio == 4 { 2 } else { 1 };
                req!(
                    p("attn_compressor_kv.weight"),
                    [hidden, coff * kv_len],
                    TensorRole::Compressor,
                    false,
                );
                req!(
                    p("attn_compressor_gate.weight"),
                    [hidden, coff * kv_len],
                    TensorRole::Compressor,
                    ratio <= 1,
                );
                req!(
                    p("attn_compressor_norm.weight"),
                    [kv_len],
                    TensorRole::Norm,
                    false,
                );
                expected.insert(p("attn_compressor_ape.weight"));
            }
            if cfg.is_index_source(layer) {
                let ih = cfg.attention.indexer.head_count as u64;
                let id = cfg.attention.indexer.key_length as u64;
                req!(p("indexer.proj.weight"), [hidden, ih], TensorRole::Indexer, false,);
                req!(
                    p("indexer.attn_q_b.weight"),
                    [q_lora, ih * id],
                    TensorRole::Indexer,
                    false,
                );
                if cfg.is_kv_source(layer) {
                    req!(
                        p("indexer.attn_k.weight"),
                        [kv_len, id],
                        TensorRole::Indexer,
                        false,
                    );
                    req!(p("indexer.k_norm.weight"), [id], TensorRole::Norm, false,);
                }
            }
            if let Some(engram) = &cfg.engram {
                if engram.layers.iter().any(|&l| l as usize == layer) {
                    let n_hash_cols = (engram.max_ngram_size as u64 - 1) * engram.heads_per_ngram as u64;
                    req!(p("engram_gate.weight"), [hidden, hc], TensorRole::Engram, false,);
                    req!(
                        p("engram_kv.weight"),
                        [n_hash_cols * engram.head_dim as u64, hidden * (hc + 1)],
                        TensorRole::Engram,
                        false,
                    );
                }
            }
            if cfg.experts.count == 0 {
                continue;
            }
        }

        // Optional V4/V4.1 tensors that may appear in compatible files.
        for name in ["hc_head_fn.weight", "hc_head_base.weight", "hc_head_scale.weight"] {
            expected.insert(name.to_string());
        }
        for layer in 0..cfg.block_count {
            expected.insert(format!("blk.{layer}.attn_compressor_ape.weight"));
            expected.insert(format!("blk.{layer}.ffn_gate_tid2eid.weight"));
        }

        for t in &self.tensors {
            if !t.ggml_type.is_known() {
                report.error(
                    "quant.unknown",
                    &t.name,
                    &format!("unknown GGML type {}", t.ggml_type.0),
                    "refusing to guess block layout; extend rustmoe-quant first",
                );
            } else if !t.ggml_type.has_reference_dequant() && options.require_decodable {
                report.error(
                    "quant.undecodable",
                    &t.name,
                    &format!("{} has no reference dequantizer in this build", t.ggml_type),
                    "use a verified quantization from docs/supported-formats.md",
                );
            }
            if !expected.contains(&t.name) && t.role != TensorRole::Unknown {
                report.warn(
                    "catalog.extra_tensor",
                    &t.name,
                    "tensor is not part of the validated DeepSeek-V4.1 catalog",
                    "new model revisions may add tensors; review before enabling",
                );
            } else if t.role == TensorRole::Unknown {
                report.warn(
                    "catalog.unknown_tensor",
                    &t.name,
                    "unclassified tensor",
                    "add a role and shape expectation or confirm it is unused",
                );
            }
        }

        // Verify per-layer expert sets are internally consistent.
        for layer in 0..cfg.block_count {
            match self.expert_layout(layer) {
                Some(layout) => {
                    if layout.expert_count != cfg.experts.count as u64 {
                        report.error(
                            "experts.count",
                            &format!("blk.{layer}.ffn_*_exps"),
                            &format!(
                                "expert count {} does not match metadata {}",
                                layout.expert_count, cfg.experts.count
                            ),
                            "the file is corrupt or the architecture does not match",
                        );
                    }
                }
                None if !self.tensors.is_empty() => report.error(
                    "experts.layout",
                    &format!("blk.{layer}.ffn_*_exps"),
                    "could not build a complete gate/down/up expert layout",
                    "check the three expert tensor shapes and types",
                ),
                None => {}
            }
        }

        if report.errors.is_empty() && cfg.tokenizer.vocab_size != vocab {
            report.error(
                "tokenizer.vocab",
                "tokenizer.ggml.tokens",
                "vocabulary size changed during parsing",
                "this should be unreachable; report a parser bug",
            );
        }
        report.stats.tensors = self.tensors.len();
        report.stats.bytes = self.summary.total_tensor_bytes;
        report.stats.roles = self.summary.roles.clone();
    }
}

#[derive(Clone, Debug)]
pub struct ExpertLayout {
    pub layer: usize,
    pub expert_count: u64,
    pub gate_bytes: u64,
    pub down_bytes: u64,
    pub up_bytes: u64,
    pub expert_bytes: u64,
    pub gate_offset: u64,
    pub down_offset: u64,
    pub up_offset: u64,
    pub gate_shard: usize,
    pub down_shard: usize,
    pub up_shard: usize,
}

#[derive(Clone, Debug)]
pub struct ValidationOptions {
    pub require_decodable: bool,
    pub require_expected_tensors: bool,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            require_decodable: true,
            require_expected_tensors: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Clone, Debug, Serialize)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub code: String,
    pub target: String,
    pub message: String,
    pub hint: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ValidationStats {
    pub tensors: usize,
    pub bytes: u64,
    pub roles: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ValidationReport {
    pub errors: Vec<ValidationIssue>,
    pub warnings: Vec<ValidationIssue>,
    pub stats: ValidationStats,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn error(&mut self, code: &str, target: &str, message: &str, hint: &str) {
        self.errors.push(ValidationIssue {
            severity: Severity::Error,
            code: code.into(),
            target: target.into(),
            message: message.into(),
            hint: Some(hint.into()),
        });
    }

    pub fn warn(&mut self, code: &str, target: &str, message: &str, hint: &str) {
        self.warnings.push(ValidationIssue {
            severity: Severity::Warning,
            code: code.into(),
            target: target.into(),
            message: message.into(),
            hint: Some(hint.into()),
        });
    }

    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

#[allow(clippy::too_many_arguments)]
fn expect(
    report: &mut ValidationReport,
    map: &BTreeMap<&str, &TensorDescriptor>,
    expected: &mut BTreeSet<String>,
    name: &str,
    dims: &[u64],
    role: TensorRole,
    optional: bool,
    options: &ValidationOptions,
) {
    expected.insert(name.to_string());
    match map.get(name) {
        None => {
            if !optional && options.require_expected_tensors {
                report.error(
                    "catalog.missing_tensor",
                    name,
                    "required tensor is missing",
                    "check that all split shards belong to the same model revision",
                );
            }
        }
        Some(t) => {
            if t.dims != dims {
                report.error(
                    "catalog.shape",
                    name,
                    &format!("shape {:?} does not match expected {:?}", t.dims, dims),
                    "do not coerce shapes; use the matching model revision",
                );
            }
            if t.role != role {
                report.warn(
                    "catalog.role",
                    name,
                    &format!("classified as {:?}, expected {:?}", t.role, role),
                    "review the classifier",
                );
            }
            if let Some(nbytes) = t.nbytes {
                let _ = nbytes;
            }
        }
    }
}

fn summarize(config: &DeepSeekV41Config, tensors: &[TensorDescriptor]) -> ModelSummary {
    let mut s = ModelSummary {
        tensor_count: tensors.len(),
        ..ModelSummary::default()
    };
    let mut roles = BTreeMap::new();
    let mut quant = BTreeMap::new();
    let mut layer_map: BTreeMap<usize, LayerSummary> = BTreeMap::new();
    for layer in 0..config.block_count {
        layer_map.insert(
            layer,
            LayerSummary {
                layer,
                compression_ratio: config.compression_ratio(layer),
                is_kv_source: config.is_kv_source(layer),
                is_index_source: config.is_index_source(layer),
                has_engram: config.is_engram_layer(layer),
                spine_bytes: 0,
                routed_expert_bytes: 0,
                shared_expert_bytes: 0,
                expert_tensor_bytes: 0,
                expert_count: None,
            },
        );
    }

    for t in tensors {
        let bytes = t.nbytes.unwrap_or(0);
        s.total_tensor_bytes = s.total_tensor_bytes.saturating_add(bytes);
        *roles.entry(t.role.as_str().to_string()).or_insert(0) += bytes;
        let q = quant
            .entry(t.ggml_type.name().to_string())
            .or_insert(TypeBytes::default());
        q.tensors += 1;
        q.bytes = q.bytes.saturating_add(bytes);
        match t.role {
            TensorRole::RoutedExpert => {
                s.routed_expert_bytes = s.routed_expert_bytes.saturating_add(bytes);
                if let Some(layer) = t.layer {
                    if let Some(l) = layer_map.get_mut(&layer) {
                        l.routed_expert_bytes = l.routed_expert_bytes.saturating_add(bytes);
                        l.expert_tensor_bytes = l.expert_tensor_bytes.saturating_add(bytes);
                        l.expert_count = t.expert_count.or(l.expert_count);
                    }
                }
            }
            TensorRole::SharedExpert => {
                s.shared_expert_bytes = s.shared_expert_bytes.saturating_add(bytes);
                if let Some(l) = t.layer.and_then(|i| layer_map.get_mut(&i)) {
                    l.shared_expert_bytes = l.shared_expert_bytes.saturating_add(bytes);
                }
            }
            TensorRole::PleTable => {
                s.ple_table_bytes = s.ple_table_bytes.saturating_add(bytes);
            }
            TensorRole::Unknown | TensorRole::Auxiliary => {
                s.auxiliary_bytes = s.auxiliary_bytes.saturating_add(bytes);
            }
            _ => {
                s.dense_spine_bytes = s.dense_spine_bytes.saturating_add(bytes);
                if let Some(l) = t.layer.and_then(|i| layer_map.get_mut(&i)) {
                    l.spine_bytes = l.spine_bytes.saturating_add(bytes);
                }
            }
        }
    }

    s.roles = roles;
    s.quant_types = quant;
    let n_expert = config.experts.count as u64;
    s.expert_count_per_layer = n_expert;
    if n_expert > 0 {
        s.avg_expert_bytes = s.routed_expert_bytes / (config.block_count as u64 * n_expert);
    }
    let mut mins = u64::MAX;
    let mut maxs = 0u64;
    for l in layer_map.values() {
        if l.expert_count.unwrap_or(0) == 0 {
            continue;
        }
        let total = l.expert_tensor_bytes;
        if total > 0 {
            let per = total / l.expert_count.unwrap_or(1);
            mins = mins.min(per);
            maxs = maxs.max(per);
        }
    }
    if mins != u64::MAX {
        s.min_expert_bytes = mins;
        s.max_expert_bytes = maxs;
    }
    s.layers = layer_map.into_values().collect();
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmoe_gguf::MetadataValue;

    fn base_metadata() -> Metadata {
        let mut m = Metadata::new();
        let mut set = |k: &str, v: MetadataValue| {
            m.insert(k.to_string(), v);
        };
        set("general.architecture", MetadataValue::String("deepseek41".into()));
        set("general.name", MetadataValue::String("test".into()));
        set("deepseek41.block_count", MetadataValue::U32(2));
        set("deepseek41.context_length", MetadataValue::U32(8192));
        set("deepseek41.embedding_length", MetadataValue::U32(5120));
        set("deepseek41.attention.head_count", MetadataValue::U32(64));
        set("deepseek41.attention.head_count_kv", MetadataValue::U32(1));
        set("deepseek41.attention.key_length", MetadataValue::U32(512));
        set("deepseek41.attention.value_length", MetadataValue::U32(512));
        set("deepseek41.attention.q_lora_rank", MetadataValue::U32(1280));
        set("deepseek41.attention.sliding_window", MetadataValue::U32(128));
        set("deepseek41.attention.output_group_count", MetadataValue::U32(8));
        set("deepseek41.attention.output_lora_rank", MetadataValue::U32(1024));
        set("deepseek41.attention.indexer.head_count", MetadataValue::U32(32));
        set("deepseek41.attention.indexer.key_length", MetadataValue::U32(128));
        set("deepseek41.attention.indexer.top_k", MetadataValue::U32(512));
        set(
            "deepseek41.attention.compress_ratios",
            MetadataValue::Array(vec![MetadataValue::U32(0), MetadataValue::U32(2)]),
        );
        set(
            "deepseek41.attention.kv_source_layers",
            MetadataValue::Array(vec![MetadataValue::U32(1)]),
        );
        set(
            "deepseek41.attention.indexer.source_layers",
            MetadataValue::Array(vec![MetadataValue::U32(1)]),
        );
        set("deepseek41.hyper_connection.count", MetadataValue::U32(4));
        set("deepseek41.expert_count", MetadataValue::U32(384));
        set("deepseek41.expert_used_count", MetadataValue::U32(6));
        set("deepseek41.expert_shared_count", MetadataValue::U32(1));
        set("deepseek41.expert_feed_forward_length", MetadataValue::U32(2304));
        set("deepseek41.expert_gating_func", MetadataValue::U32(4));
        set("deepseek41.expert_weights_scale", MetadataValue::F32(1.5));
        set("deepseek41.expert_weights_norm", MetadataValue::Bool(true));
        set("deepseek41.rope.dimension_count", MetadataValue::U32(64));
        set("deepseek41.rope.freq_base", MetadataValue::F32(10000.0));
        set(
            "deepseek41.attention.layer_norm_rms_epsilon",
            MetadataValue::F32(1e-20),
        );
        set(
            "deepseek41.ple.layers",
            MetadataValue::Array(vec![MetadataValue::U32(1)]),
        );
        set("deepseek41.ple.ngram_size", MetadataValue::U32(4));
        set("deepseek41.ple.heads_per_ngram", MetadataValue::U32(8));
        set(
            "deepseek41.embedding_length_per_layer_input",
            MetadataValue::U32(256),
        );
        set(
            "deepseek41.ple.head_vocab_sizes",
            MetadataValue::Array(vec![MetadataValue::U64(100), MetadataValue::U64(200)]),
        );
        set("deepseek41.engram.vocab_size", MetadataValue::U32(1000));
        set("deepseek41.engram.compressed_vocab_size", MetadataValue::U32(100));
        set(
            "tokenizer.ggml.tokens",
            MetadataValue::Array(vec![
                MetadataValue::String("a".into()),
                MetadataValue::String("b".into()),
            ]),
        );
        set("tokenizer.ggml.bos_token_id", MetadataValue::U32(0));
        set("tokenizer.ggml.eos_token_id", MetadataValue::U32(1));
        m
    }

    #[test]
    fn parses_deepseek_facts() {
        let cfg = DeepSeekV41Config::from_metadata(&base_metadata()).unwrap();
        assert_eq!(cfg.block_count, 2);
        assert_eq!(cfg.experts.count, 384);
        assert_eq!(cfg.experts.used_count, 6);
        assert_eq!(cfg.experts.gating_function, "sqrtsoftplus");
        assert_eq!(cfg.attention.key_length, 512);
        assert_eq!(cfg.q_head_width(), 32768);
        assert_eq!(cfg.o_width(), 8192);
        assert_eq!(cfg.engram.unwrap().table_rows, 300);
    }

    #[test]
    fn rejects_arch_mismatch() {
        let mut m = base_metadata();
        m.insert(
            "general.architecture",
            MetadataValue::String("deepseek_v41".into()),
        );
        let err = DeepSeekV41Config::from_metadata(&m).unwrap_err();
        assert!(matches!(err, ModelError::UnsupportedArchitecture(_)));
    }

    #[test]
    fn classifies_tensor_roles() {
        assert_eq!(
            classify_tensor("blk.3.ffn_gate_exps.weight"),
            TensorRole::RoutedExpert
        );
        assert_eq!(
            classify_tensor("blk.3.ffn_gate_shexp.weight"),
            TensorRole::SharedExpert
        );
        assert_eq!(classify_tensor("blk.3.ffn_gate_inp.weight"), TensorRole::Router);
        assert_eq!(
            classify_tensor("blk.3.hc_attn_fn.weight"),
            TensorRole::HyperConnection
        );
        assert_eq!(
            classify_tensor("blk.3.attn_compressor_kv.weight"),
            TensorRole::Compressor
        );
        assert_eq!(layer_index("blk.39.attn_norm.weight"), Some(39));
    }
}
