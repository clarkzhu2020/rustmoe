//! Automatic llama.cpp placement for the available device.
//!
//! The policy is intentionally conservative: use reported free VRAM, reserve
//! compute/safety/KV first, place dense weights next, then give the rest to
//! complete expert layers and finally to the hot-expert cache.

use crate::{HardwareProfile, MemoryError};
use rustmoe_attention::{CacheDtype, KvPlan};
use rustmoe_model::{DeepSeekV41Config, ModelSummary};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct AutoDevicePlan {
    pub gpu_name: String,
    pub free_vram: u64,
    pub total_vram: u64,
    pub available_ram: u64,
    pub context: u64,
    pub n_gpu_layers: i32,
    pub cpu_moe_layers: i32,
    pub gpu_expert_layers: u32,
    pub moe_cache_mib: i32,
    pub n_batch: i32,
    pub threads: i32,
    pub dense_gpu_bytes: u64,
    pub expert_layer_bytes: u64,
    pub predicted_vram_bytes: u64,
    pub predicted_headroom: u64,
    pub kv_bytes: u64,
    pub notes: Vec<String>,
}

pub fn auto_llama_plan(
    config: &DeepSeekV41Config,
    summary: &ModelSummary,
    hardware: &HardwareProfile,
    requested_context: Option<u64>,
) -> Result<AutoDevicePlan, MemoryError> {
    if summary.tensor_count == 0 || summary.routed_expert_bytes == 0 {
        return Err(MemoryError::NoCatalog);
    }
    let gpu = hardware.selected_gpu().ok_or(MemoryError::NoGpu)?;
    if gpu.free_vram < 6 * 1024 * 1024 * 1024 {
        return Err(MemoryError::NotEnoughVram {
            free: gpu.free_vram,
            need: 6 * 1024 * 1024 * 1024,
        });
    }

    let context = match requested_context {
        Some(c) if c > 0 => c,
        _ => auto_context(gpu.free_vram, hardware.available_ram, config.context_length),
    };
    if context > config.context_length {
        return Err(MemoryError::ContextTooLong {
            requested: context,
            limit: config.context_length,
        });
    }

    let kv_bytes = KvPlan::estimate(config, context, CacheDtype::F16).total_bytes;
    let compute_reserve = 2048 * 1024 * 1024;
    let safety = 1024 * 1024 * 1024;
    let available = gpu
        .free_vram
        .saturating_sub(compute_reserve)
        .saturating_sub(safety)
        .saturating_sub(kv_bytes);
    let dense_gpu = summary.dense_spine_bytes.saturating_mul(120) / 100;
    let expert_layer =
        (summary.routed_expert_bytes / config.block_count.max(1) as u64).saturating_mul(135) / 100;

    let mut notes = Vec::new();
    notes.push(format!(
        "selected {} with {} free of {}",
        gpu.name,
        crate::format_bytes(gpu.free_vram),
        crate::format_bytes(gpu.total_vram)
    ));
    notes.push(format!(
        "context {} selected for {} free VRAM and {} available RAM",
        context,
        crate::format_bytes(gpu.free_vram),
        crate::format_bytes(hardware.available_ram)
    ));
    if hardware.available_ram < 32 * 1024 * 1024 * 1024 {
        notes.push("less than 32 GiB host RAM available; expect mmap page-cache pressure".into());
    }

    if available <= dense_gpu {
        notes.push(
            "free VRAM cannot hold the dense spine plus reserves; automatic mode keeps all experts on the host".into(),
        );
    }
    let gpu_expert_layers = if available > dense_gpu && expert_layer > 0 {
        ((available - dense_gpu) / expert_layer).min(config.block_count as u64) as u32
    } else {
        0
    };
    let after_layers = available
        .saturating_sub(dense_gpu.saturating_add(expert_layer.saturating_mul(gpu_expert_layers as u64)));
    let mut cache = (after_layers / 2).min(4096 * 1024 * 1024);
    if cache < 256 * 1024 * 1024 {
        cache = 0;
    }
    let moe_cache_mib = (cache / (1024 * 1024)) as i32;
    let cpu_moe_layers = config.block_count.saturating_sub(gpu_expert_layers as usize) as i32;
    let predicted = dense_gpu
        .saturating_add(expert_layer.saturating_mul(gpu_expert_layers as u64))
        .saturating_add(cache)
        .saturating_add(kv_bytes)
        .saturating_add(compute_reserve)
        .saturating_add(safety);
    let headroom = gpu.free_vram.saturating_sub(predicted);

    notes.push(format!(
        "{} expert layers on GPU, {} on host, {} MiB hot-expert cache",
        gpu_expert_layers, cpu_moe_layers, moe_cache_mib
    ));
    if gpu_expert_layers == 0 {
        notes.push("all routed experts stay on the host for this device".into());
    }
    if hardware.available_ram < 20 * 1024 * 1024 * 1024 {
        notes.push("host RAM is tight; keep mmap and lazy loading enabled".into());
    }

    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(16)
        .clamp(4, 32);

    Ok(AutoDevicePlan {
        gpu_name: gpu.name.clone(),
        free_vram: gpu.free_vram,
        total_vram: gpu.total_vram,
        available_ram: hardware.available_ram,
        context,
        n_gpu_layers: 999,
        cpu_moe_layers,
        gpu_expert_layers,
        moe_cache_mib,
        n_batch: 512,
        threads,
        dense_gpu_bytes: dense_gpu,
        expert_layer_bytes: expert_layer,
        predicted_vram_bytes: predicted,
        predicted_headroom: headroom,
        kv_bytes,
        notes,
    })
}

fn auto_context(free_vram: u64, available_ram: u64, model_limit: u64) -> u64 {
    let gib = 1024 * 1024 * 1024u64;
    let mut context = if free_vram >= 28 * gib {
        16384
    } else if free_vram >= 20 * gib {
        8192
    } else {
        4096
    };
    if available_ram < 32 * gib {
        context = context.min(4096);
    }
    context.min(model_limit)
}
