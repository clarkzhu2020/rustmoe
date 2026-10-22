//! Command-line interface for RustMoE.

use crate::agent::AgentConfig;
use crate::api;
use crate::progress::LoadingProgress;
use crate::runtime::{BackendKind, LlamaRuntimeOptions, Runtime, RuntimeOptions};
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use rustmoe_attention::{format_bytes, CacheDtype};
use rustmoe_gguf::{GgufFile, GgufSet};
use rustmoe_inference::GenerationConfig;
use rustmoe_memory::{plan, BudgetRequest, Diagnostic, HardwareProfile, PlanInput, PlanStatus, Severity};
use rustmoe_model::{DeepSeekV41Config, ModelDescriptor, ValidationOptions};
use rustmoe_tokenizer::BpeTokenizer;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "rustmoe",
    version,
    about = "RustMoE: low-VRAM GGUF runtime with explicit expert tiering"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Inspect GGUF metadata, tensor catalog, and quantization facts.
    Inspect {
        model: PathBuf,
        /// Parse only the given shard. Useful for a metadata-only shard.
        #[arg(long)]
        single_file: bool,
        #[arg(long)]
        json: bool,
        /// Print every tensor descriptor.
        #[arg(long)]
        tensors: bool,
    },
    /// Validate split continuity, tensor shapes, and quantization support.
    Validate {
        model: PathBuf,
        #[arg(long)]
        json: bool,
        /// Downgrade missing native dequantizers to warnings.
        #[arg(long)]
        allow_undecodable: bool,
    },
    /// Compute a VRAM, RAM, and NVMe budget before loading weights.
    Plan {
        model: PathBuf,
        #[arg(long, default_value_t = 8192)]
        context: u64,
        #[arg(long, default_value = "f16", value_parser = parse_cache_dtype)]
        cache_dtype: CacheDtype,
        #[arg(long, value_parser = parse_size)]
        vram: Option<u64>,
        #[arg(long, value_parser = parse_size)]
        ram: Option<u64>,
        #[arg(long, value_parser = parse_size)]
        disk: Option<u64>,
        #[arg(long, value_parser = parse_size)]
        gpu_cache: Option<u64>,
        #[arg(long, value_parser = parse_size)]
        host_cache: Option<u64>,
        #[arg(long, default_value_t = 2048)]
        prefill: u64,
        #[arg(long, value_parser = parse_size)]
        workspace: Option<u64>,
        #[arg(long)]
        host_spine: bool,
        #[arg(long)]
        json: bool,
    },
    /// Tokenize text with the GGUF vocabulary.
    Tokenize {
        model: PathBuf,
        #[arg(long)]
        single_file: bool,
        #[arg(long)]
        text: String,
        #[arg(long)]
        json: bool,
    },
    /// Run a deterministic simulation of the decode runtime.
    Simulate {
        model: PathBuf,
        /// Load one shard; only valid for simulation and metadata inspection.
        #[arg(long)]
        single_file: bool,
        #[arg(long, default_value = "simulation", value_parser = parse_backend)]
        backend: BackendKind,
        /// Disable automatic placement for the real backend.
        #[arg(long)]
        no_auto_device: bool,
        #[arg(long)]
        gpu_layers: Option<i32>,
        #[arg(long)]
        cpu_moe_layers: Option<i32>,
        #[arg(long)]
        moe_cache_mib: Option<i32>,
        #[arg(long, default_value_t = 512)]
        llama_batch: i32,
        #[arg(long, default_value_t = 16)]
        threads: i32,
        #[arg(long, default_value_t = 1)]
        load_mode: i32,
        #[arg(long, default_value = "on", value_parser = parse_lazy_mode)]
        lazy_mode: i32,
        #[arg(long, default_value = "Hello from RustMoE")]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        /// 0 selects automatically from the detected device.
        #[arg(long, default_value_t = 0)]
        context: u64,
        #[arg(long, default_value_t = 0xD5E14A1110172026)]
        seed: u64,
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        #[arg(long, default_value_t = 1.0)]
        repetition_penalty: f32,
        #[arg(long, value_parser = parse_size)]
        gpu_cache: Option<u64>,
        #[arg(long, value_parser = parse_size)]
        host_cache: Option<u64>,
        #[arg(long, default_value_t = 2048)]
        prefill_chunk: usize,
        #[arg(long)]
        json: bool,
    },
    /// Serve an OpenAI-compatible API backed by the simulation runtime.
    Serve {
        model: PathBuf,
        /// Load one shard; only valid for simulation and metadata inspection.
        #[arg(long)]
        single_file: bool,
        #[arg(long, default_value = "simulation", value_parser = parse_backend)]
        backend: BackendKind,
        /// Disable automatic placement for the real backend.
        #[arg(long)]
        no_auto_device: bool,
        #[arg(long)]
        gpu_layers: Option<i32>,
        #[arg(long)]
        cpu_moe_layers: Option<i32>,
        #[arg(long)]
        moe_cache_mib: Option<i32>,
        #[arg(long, default_value_t = 512)]
        llama_batch: i32,
        #[arg(long, default_value_t = 16)]
        threads: i32,
        #[arg(long, default_value_t = 1)]
        load_mode: i32,
        #[arg(long, default_value = "on", value_parser = parse_lazy_mode)]
        lazy_mode: i32,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// 0 selects automatically from the detected device.
        #[arg(long, default_value_t = 0)]
        context: u64,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long, value_parser = parse_size)]
        gpu_cache: Option<u64>,
        #[arg(long, value_parser = parse_size)]
        host_cache: Option<u64>,
        #[arg(long, default_value_t = 2048)]
        prefill_chunk: usize,
        #[arg(long)]
        model_id: Option<String>,
        #[arg(long, default_value_t = 2)]
        prefetch_layers: usize,
        /// Root directory the development agent may access.
        #[arg(long, default_value = "workspace")]
        agent_root: PathBuf,
        /// Allow the agent to run shell commands (pwsh) inside the root.
        #[arg(long)]
        agent_allow_shell: bool,
        #[arg(long, default_value_t = 24)]
        agent_max_steps: usize,
    },
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();
    init_tracing();
    match cli.command {
        Command::Inspect {
            model,
            single_file,
            json,
            tensors,
        } => inspect(&model, single_file, json, tensors),
        Command::Validate {
            model,
            json,
            allow_undecodable,
        } => validate(&model, json, allow_undecodable),
        Command::Plan {
            model,
            context,
            cache_dtype,
            vram,
            ram,
            disk,
            gpu_cache,
            host_cache,
            prefill,
            workspace,
            host_spine,
            json,
        } => run_plan(
            &model,
            context,
            cache_dtype,
            vram,
            ram,
            disk,
            gpu_cache,
            host_cache,
            prefill,
            workspace,
            host_spine,
            json,
        ),
        Command::Tokenize {
            model,
            single_file,
            text,
            json,
        } => run_tokenize(&model, single_file, &text, json),
        Command::Simulate {
            model,
            single_file,
            backend,
            no_auto_device,
            gpu_layers,
            cpu_moe_layers,
            moe_cache_mib,
            llama_batch,
            threads,
            load_mode,
            lazy_mode,
            prompt,
            max_tokens,
            context,
            seed,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            gpu_cache,
            host_cache,
            prefill_chunk,
            json,
        } => {
            run_simulate(
                &model,
                single_file,
                backend,
                LlamaRuntimeOptions {
                    n_gpu_layers: gpu_layers,
                    cpu_moe_layers,
                    moe_cache_mib,
                    n_batch: llama_batch,
                    n_threads: threads,
                    n_threads_batch: threads,
                    load_mode,
                    lazy_mode,
                },
                &prompt,
                max_tokens,
                context,
                !no_auto_device,
                seed,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                gpu_cache,
                host_cache,
                prefill_chunk,
                json,
            )
            .await
        }
        Command::Serve {
            model,
            single_file,
            backend,
            no_auto_device,
            gpu_layers,
            cpu_moe_layers,
            moe_cache_mib,
            llama_batch,
            threads,
            load_mode,
            lazy_mode,
            addr,
            context,
            api_key,
            gpu_cache,
            host_cache,
            prefill_chunk,
            model_id,
            prefetch_layers,
            agent_root,
            agent_allow_shell,
            agent_max_steps,
        } => {
            serve(
                model,
                single_file,
                backend,
                LlamaRuntimeOptions {
                    n_gpu_layers: gpu_layers,
                    cpu_moe_layers,
                    moe_cache_mib,
                    n_batch: llama_batch,
                    n_threads: threads,
                    n_threads_batch: threads,
                    load_mode,
                    lazy_mode,
                },
                addr,
                context,
                !no_auto_device,
                api_key,
                gpu_cache,
                host_cache,
                prefill_chunk,
                model_id,
                prefetch_layers,
                AgentConfig {
                    root: agent_root,
                    allow_shell: agent_allow_shell,
                    max_steps: agent_max_steps,
                },
            )
            .await
        }
    }
}

fn inspect(model: &Path, single_file: bool, json: bool, tensors: bool) -> Result<()> {
    if single_file {
        let file = GgufFile::parse_path(model, &Default::default())
            .with_context(|| format!("parse {}", model.display()))?;
        let config = DeepSeekV41Config::from_metadata(&file.metadata).ok();
        if json {
            let mut metadata = serde_json::Map::new();
            for (k, v) in file.metadata.iter() {
                if k.starts_with("tokenizer.ggml.tokens")
                    || k.starts_with("tokenizer.ggml.merges")
                    || k.starts_with("tokenizer.ggml.token_type")
                    || k == "deepseek41.engram.token_map"
                {
                    continue;
                }
                metadata.insert(k.clone(), serde_json::to_value(v)?);
            }
            let body = json!({
                "mode": "single_file",
                "path": file.path,
                "version": file.version,
                "header_bytes": file.header_bytes,
                "data_offset": file.data_offset,
                "file_len": file.file_len,
                "tensor_count": file.tensors.len(),
                "split": file.split,
                "config": config,
                "metadata": metadata,
            });
            println!("{}", serde_json::to_string_pretty(&body)?);
        } else {
            println!("model:       {}", file.path.display());
            println!("mode:        single shard (split completeness not checked)");
            println!(
                "GGUF:        v{} header={} data={} file={}",
                file.version,
                format_bytes(file.header_bytes),
                format_bytes(file.data_offset),
                format_bytes(file.file_len)
            );
            if let Some(split) = &file.split {
                println!(
                    "split:       shard {} of {} ({} total tensors)",
                    split.no + 1,
                    split.count,
                    split.tensor_count.unwrap_or(0)
                );
            }
            println!("tensors:     {}", file.tensors.len());
            if let Some(c) = &config {
                print_config(c);
            }
            for t in &file.tensors {
                println!("  {:<40} {:>10} {:?}", t.name, t.ggml_type, t.dims);
            }
        }
        return Ok(());
    }

    let descriptor = ModelDescriptor::load(model).with_context(|| format!("load {}", model.display()))?;
    if json {
        let body = json!({
            "path": descriptor.path,
            "config": descriptor.config,
            "summary": descriptor.summary,
            "shards": descriptor.gguf.files.iter().map(|f| json!({
                "path": f.path,
                "version": f.version,
                "split_no": f.split.as_ref().map(|s| s.no),
                "split_count": f.split.as_ref().map(|s| s.count),
                "header_bytes": f.header_bytes,
                "data_offset": f.data_offset,
                "file_len": f.file_len,
                "tensors": f.tensors.len(),
            })).collect::<Vec<_>>(),
            "tensors": if tensors { serde_json::to_value(&descriptor.tensors)? } else { json!([]) },
        });
        println!("{}", serde_json::to_string_pretty(&body)?);
    } else {
        println!("model:       {}", descriptor.path.display());
        println!("architecture:{}", descriptor.config.architecture);
        print_config(&descriptor.config);
        println!();
        println!("shards:      {}", descriptor.gguf.files.len());
        for f in &descriptor.gguf.files {
            let split = f
                .split
                .as_ref()
                .map(|s| format!("{}/{}", s.no + 1, s.count))
                .unwrap_or_else(|| "single".into());
            println!(
                "  {:<62} {:>12}  split {}  {} tensors",
                f.path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                format_bytes(f.file_len),
                split,
                f.tensors.len()
            );
        }
        println!();
        println!("tensors:     {}", descriptor.summary.tensor_count);
        println!(
            "tensor bytes:{}",
            format_bytes(descriptor.summary.total_tensor_bytes)
        );
        println!(
            "routed experts: {} ({})",
            format_bytes(descriptor.summary.routed_expert_bytes),
            descriptor.summary.expert_count_per_layer
        );
        println!(
            "dense spine: {}  shared: {}  PLE table: {}",
            format_bytes(descriptor.summary.dense_spine_bytes),
            format_bytes(descriptor.summary.shared_expert_bytes),
            format_bytes(descriptor.summary.ple_table_bytes)
        );
        println!("quant types:");
        for (name, t) in &descriptor.summary.quant_types {
            println!(
                "  {:<8} {:>5} tensors  {}",
                name,
                t.tensors,
                format_bytes(t.bytes)
            );
        }
        if tensors {
            println!("\ntensor directory:");
            for t in &descriptor.tensors {
                println!(
                    "  {:<42} {:>9} {:>18} {:?}",
                    t.name,
                    t.ggml_type.to_string(),
                    format_bytes(t.nbytes.unwrap_or(0)),
                    t.dims
                );
            }
        }
    }
    Ok(())
}

fn print_config(c: &DeepSeekV41Config) {
    println!("name:        {}", c.name.as_deref().unwrap_or("-"));
    println!(
        "layers:      {}  hidden: {}  context: {}",
        c.block_count, c.hidden_size, c.context_length
    );
    println!(
        "attention:   MLA q_lora={} kv={} rope={} heads={} groups={} window={}",
        c.attention.q_lora_rank,
        c.attention.key_length,
        c.rope.dimension_count,
        c.attention.head_count,
        c.attention.output_group_count,
        c.attention.sliding_window
    );
    println!(
        "experts:     {} routed, top-{} + {} shared, ffn={}  gating={}  scale={}",
        c.experts.count,
        c.experts.used_count,
        c.experts.shared_count,
        c.experts.intermediate_size,
        c.experts.gating_function,
        c.experts.weights_scale
    );
    println!(
        "compress:    {:?}  kv sources={:?}  index sources={:?}",
        c.compress_ratios, c.kv_source_layers, c.index_source_layers
    );
    if let Some(e) = &c.engram {
        println!(
            "engram/PLE:  layers={:?} rows={} head_dim={} ngram={}",
            e.layers, e.table_rows, e.head_dim, e.max_ngram_size
        );
    }
    if let Some(d) = &c.dspark {
        println!(
            "dspark:      block={} experts={} top-{} target={:?} (separate drafter)",
            d.block_size, d.expert_count, d.expert_used_count, d.target_layers
        );
    }
    println!(
        "tokenizer:   vocab={} model={} pre={} eos={:?}",
        c.tokenizer.vocab_size,
        c.tokenizer.model.as_deref().unwrap_or("-"),
        c.tokenizer.pre_tokenizer.as_deref().unwrap_or("-"),
        c.tokenizer.eos_token_id
    );
    println!(
        "quant meta:  file_type={:?} version={:?}",
        c.quantization.file_type, c.quantization.quantization_version
    );
}

fn validate(model: &Path, json: bool, allow_undecodable: bool) -> Result<()> {
    let descriptor = ModelDescriptor::load(model).with_context(|| format!("load {}", model.display()))?;
    let options = ValidationOptions {
        require_decodable: !allow_undecodable,
        require_expected_tensors: true,
    };
    let report = descriptor.validate_with(&options);
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("model: {}", descriptor.path.display());
        println!(
            "tensors: {}  bytes: {}  errors: {}  warnings: {}",
            report.stats.tensors,
            format_bytes(report.stats.bytes),
            report.errors.len(),
            report.warnings.len()
        );
        for issue in report.errors.iter().chain(report.warnings.iter()) {
            let tag = match issue.severity {
                rustmoe_model::Severity::Error => "ERROR",
                rustmoe_model::Severity::Warning => "WARN ",
            };
            println!("  {} [{}] {}: {}", tag, issue.code, issue.target, issue.message);
            if let Some(hint) = &issue.hint {
                println!("        hint: {hint}");
            }
        }
    }
    if !report.is_ok() {
        bail!("validation failed with {} error(s)", report.error_count());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_plan(
    model: &Path,
    context: u64,
    cache_dtype: CacheDtype,
    vram: Option<u64>,
    ram: Option<u64>,
    disk: Option<u64>,
    gpu_cache: Option<u64>,
    host_cache: Option<u64>,
    prefill: u64,
    workspace: Option<u64>,
    host_spine: bool,
    json: bool,
) -> Result<()> {
    let descriptor = ModelDescriptor::load(model).with_context(|| format!("load {}", model.display()))?;
    let input = PlanInput::from_model(&descriptor.config, &descriptor.summary, context, cache_dtype)
        .context("build memory plan input")?;
    let hardware = HardwareProfile::detect(Some(model));
    let mut request = BudgetRequest {
        context,
        cache_dtype,
        prefill_tokens: prefill.min(context),
        gpu_cache_bytes: gpu_cache,
        host_cache_bytes: host_cache,
        workspace_bytes: workspace,
        host_spine: host_spine.then_some(true),
        ..BudgetRequest::default()
    };
    if let Some(vram) = vram {
        request.vram_total = Some(vram);
        request.vram_free = Some(vram);
    }
    if let Some(ram) = ram {
        request.ram_total = Some(ram);
        request.ram_available = Some(ram);
    }
    request.disk_available = disk;
    let plan = plan(&input, &hardware, &request).context("compute memory plan")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }
    println!("{}", plan.summary_line());
    println!(
        "status: {}",
        match plan.status {
            PlanStatus::Compatible => "COMPATIBLE",
            PlanStatus::LowSpeed => "LOW-SPEED COMPATIBLE",
            PlanStatus::Insufficient => "INSUFFICIENT",
        }
    );
    println!();
    println!(
        "VRAM  total {}  usable {}  free-reported {}",
        format_bytes(plan.vram.total),
        format_bytes(plan.vram.usable),
        plan.vram
            .free_reported
            .map(format_bytes)
            .unwrap_or_else(|| "?".into())
    );
    println!(
        "  display/runtime/safety: {} / {} / {}",
        format_bytes(plan.vram.display_reserve),
        format_bytes(plan.vram.runtime_reserve),
        format_bytes(plan.vram.safety_reserve)
    );
    println!(
        "  spine {:?} {}  KV {}  workspace {}  expert-cache {}  headroom {}",
        plan.vram.spine_placement,
        format_bytes(plan.vram.spine),
        format_bytes(plan.vram.kv),
        format_bytes(plan.vram.workspace),
        format_bytes(plan.vram.expert_cache),
        format_bytes(plan.vram.headroom)
    );
    println!(
        "RAM   total {}  available {}  reserve {}  io {}",
        format_bytes(plan.ram.total),
        format_bytes(plan.ram.available),
        format_bytes(plan.ram.reserve),
        format_bytes(plan.ram.io_buffers)
    );
    println!(
        "  resident-spine {}  host-cache {}  headroom {}",
        format_bytes(plan.ram.resident_spine),
        format_bytes(plan.ram.host_cache),
        format_bytes(plan.ram.headroom)
    );
    println!(
        "Disk  available {}  tensor-store {}  PLE {}",
        format_bytes(plan.disk.available),
        format_bytes(plan.disk.model_tensor_bytes),
        format_bytes(plan.disk.ple_table_bytes)
    );
    println!();
    println!("{}", plan.kv.summary());
    for component in &plan.kv.components {
        println!(
            "  {:<26} layers={} rows={} row={} bytes={}",
            component.name,
            component.layer_count,
            component.rows_per_layer,
            format_bytes(component.row_bytes),
            format_bytes(component.bytes)
        );
    }
    println!();
    println!(
        "Expert I/O  avg/expert={} x {} = {} per cold token; GPU cache {} experts ({:.2}%), host {} experts ({:.2}%)",
        format_bytes(plan.expert_io.avg_expert_bytes),
        plan.expert_io.experts_per_token,
        format_bytes(plan.expert_io.cold_expert_bytes_per_token),
        plan.expert_io.gpu_cache_experts,
        plan.expert_io.gpu_cache_fraction * 100.0,
        plan.expert_io.host_cache_experts,
        plan.expert_io.host_cache_fraction * 100.0
    );
    println!(
        "  cold I/O only: {:.1} ms/token (NVMe {:.2} GB/s, PCIe {:.2} GB/s); this is not a throughput promise",
        plan.expert_io.cold_io_bound_ms_per_token,
        plan.expert_io.nvme_bandwidth as f64 / 1e9,
        plan.expert_io.pcie_bandwidth as f64 / 1e9
    );
    println!(
        "prefill chunk: {} tokens (activation {}/token)",
        plan.prefill_chunk,
        format_bytes(plan.prefill_activation_bytes_per_token)
    );
    if !plan.diagnostics.is_empty() {
        println!("\ndiagnostics:");
        for d in &plan.diagnostics {
            print_diagnostic(d);
        }
    }
    if !plan.caveats.is_empty() {
        println!("\ncaveats:");
        for c in &plan.caveats {
            println!("  - {c}");
        }
    }
    if plan.status == PlanStatus::Insufficient {
        eprintln!("\nwarning: the budget does not close; do not start the model with this configuration");
    }
    Ok(())
}

fn print_diagnostic(d: &Diagnostic) {
    let tag = match d.severity {
        Severity::Info => "INFO ",
        Severity::Warning => "WARN ",
        Severity::Error => "ERROR",
    };
    println!("  {} [{}] {}", tag, d.code, d.message);
    if let Some(hint) = &d.hint {
        println!("        hint: {hint}");
    }
}

fn load_tokenizer(model: &Path, single_file: bool) -> Result<BpeTokenizer> {
    if single_file {
        let file = GgufFile::parse_path(model, &Default::default())
            .with_context(|| format!("parse {}", model.display()))?;
        BpeTokenizer::from_metadata(&file.metadata).map_err(Into::into)
    } else {
        let gguf = GgufSet::open(model).with_context(|| format!("open {}", model.display()))?;
        BpeTokenizer::from_gguf(&gguf).map_err(Into::into)
    }
}

fn run_tokenize(model: &Path, single_file: bool, text: &str, json: bool) -> Result<()> {
    let tokenizer = load_tokenizer(model, single_file)?;
    let ids = tokenizer.encode_raw(text);
    let tokens: Vec<&str> = ids
        .iter()
        .map(|id| tokenizer.token_text(*id).unwrap_or("<invalid>"))
        .collect();
    let decoded = tokenizer.decode(&ids);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ids": ids,
                "tokens": tokens,
                "decoded": decoded,
                "pre_tokenizer": tokenizer.pre_tokenizer,
                "pre_tokenizer_exact": tokenizer.pre_tokenizer_exact,
            }))?
        );
    } else {
        println!("ids:     {:?}", ids);
        println!("tokens:  {:?}", tokens);
        println!("decoded: {:?}", decoded);
        println!(
            "pre:     {} (exact={})",
            tokenizer.pre_tokenizer, tokenizer.pre_tokenizer_exact
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_simulate(
    model: &Path,
    single_file: bool,
    backend: BackendKind,
    llama: LlamaRuntimeOptions,
    prompt: &str,
    max_tokens: usize,
    context: u64,
    auto_device: bool,
    seed: u64,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    gpu_cache: Option<u64>,
    host_cache: Option<u64>,
    prefill_chunk: usize,
    json: bool,
) -> Result<()> {
    let progress = (backend == BackendKind::LlamaCpp).then(|| LoadingProgress::start("DeepSeek-V4.1"));
    let runtime = if single_file {
        Runtime::load_single_shard(
            model,
            RuntimeOptions {
                context,
                gpu_cache_bytes: gpu_cache,
                host_cache_bytes: host_cache,
                prefill_chunk,
                seed,
                backend,
                auto_device,
                llama,
                ..RuntimeOptions::default()
            },
        )?
    } else {
        Runtime::load(
            model,
            RuntimeOptions {
                context,
                gpu_cache_bytes: gpu_cache,
                host_cache_bytes: host_cache,
                prefill_chunk,
                seed,
                backend,
                auto_device,
                llama,
                ..RuntimeOptions::default()
            },
        )?
    };
    if let Some(progress) = progress {
        progress.finish();
    }
    if !json && runtime.info.simulation {
        eprintln!("warning: backend is a deterministic simulation, not DeepSeek-V4.1 inference");
    }
    let cfg = GenerationConfig {
        max_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        seed,
        ..GenerationConfig::default()
    };
    let engine = Arc::clone(&runtime.engine);
    let prompt = prompt.to_string();
    let result = tokio::task::spawn_blocking(move || {
        let mut engine = engine.blocking_lock();
        engine.generate(&prompt, &cfg)
    })
    .await??;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("{}", result.text);
        println!();
        println!("backend: {} (simulation={})", result.backend, result.simulation);
        if let Some(plan) = &runtime.info.device_plan {
            println!(
                "auto: gpu={} free={} context={} gpu_experts={} host_experts={} cache={}MiB headroom={}",
                plan.gpu_name,
                format_bytes(plan.free_vram),
                plan.context,
                plan.gpu_expert_layers,
                plan.cpu_moe_layers,
                plan.moe_cache_mib,
                format_bytes(plan.predicted_headroom)
            );
        }
        println!(
            "prompt={} completion={} finish={:?} ttft={:.2}ms prefill={:.2}ms decode={:.2}ms total={:.2}ms",
            result.prompt_token_count,
            result.completion_token_count,
            result.finish_reason,
            result.ttft_ms,
            result.prefill_ms,
            result.decode_ms,
            result.total_ms
        );
        println!(
            "cache: gpu={} ({:.2}%) host={} ({:.2}%) disk={} read={} h2d={} batches={}",
            result.scheduler.gpu_hits,
            result.scheduler.hit_rate() * 100.0,
            result.scheduler.host_hits,
            result.scheduler.host_hits as f64 / result.scheduler.requests.max(1) as f64 * 100.0,
            result.scheduler.disk_misses,
            format_bytes(result.scheduler.disk_bytes),
            format_bytes(result.scheduler.h2d_bytes),
            result.scheduler.read_batches
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn serve(
    model: PathBuf,
    single_file: bool,
    backend: BackendKind,
    llama: LlamaRuntimeOptions,
    addr: String,
    context: u64,
    auto_device: bool,
    api_key: Option<String>,
    gpu_cache: Option<u64>,
    host_cache: Option<u64>,
    prefill_chunk: usize,
    model_id: Option<String>,
    prefetch_layers: usize,
    agent: AgentConfig,
) -> Result<()> {
    let addr: std::net::SocketAddr = addr
        .parse()
        .with_context(|| format!("invalid listen address {addr:?}"))?;
    let options = RuntimeOptions {
        context,
        gpu_cache_bytes: gpu_cache,
        host_cache_bytes: host_cache,
        prefill_chunk,
        prefetch_layers,
        model_id,
        backend,
        auto_device,
        llama,
        ..RuntimeOptions::default()
    };
    let progress = (backend == BackendKind::LlamaCpp).then(|| LoadingProgress::start("DeepSeek-V4.1"));
    let runtime = if single_file {
        Runtime::load_single_shard(&model, options)?
    } else {
        Runtime::load(&model, options)?
    }
    .with_api_key(api_key);
    if let Some(progress) = progress {
        progress.finish();
    }
    for warning in &runtime.info.warnings {
        tracing::warn!("{warning}");
    }
    if runtime.info.simulation {
        tracing::warn!(
            "serving the deterministic SIMULATION backend; this process does not execute DeepSeek-V4.1"
        );
    } else {
        tracing::warn!("serving llama.cpp reference execution backend");
    }
    std::fs::create_dir_all(&agent.root)
        .with_context(|| format!("create agent root {}", agent.root.display()))?;
    api::serve(Arc::new(runtime), addr, agent).await
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}

fn parse_size(raw: &str) -> Result<u64, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("empty size".into());
    }
    let split = raw
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(raw.len());
    let number: f64 = raw[..split]
        .parse()
        .map_err(|_| format!("invalid number in size {raw:?}"))?;
    let suffix = raw[split..].trim().to_ascii_lowercase();
    let multiplier: f64 = match suffix.as_str() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return Err(format!("unknown size suffix {suffix:?}")),
    };
    let bytes = number * multiplier;
    if !bytes.is_finite() || bytes < 0.0 || bytes > u64::MAX as f64 {
        return Err(format!("size {raw:?} is out of range"));
    }
    Ok(bytes.round() as u64)
}

fn parse_backend(raw: &str) -> Result<BackendKind, String> {
    match raw.to_ascii_lowercase().as_str() {
        "simulation" | "sim" => Ok(BackendKind::Simulation),
        "llama" | "llama.cpp" | "llamacpp" => Ok(BackendKind::LlamaCpp),
        _ => Err(format!("unsupported backend {raw:?}")),
    }
}

fn parse_lazy_mode(raw: &str) -> Result<i32, String> {
    match raw.to_ascii_lowercase().as_str() {
        "off" => Ok(0),
        "auto" => Ok(1),
        "on" => Ok(2),
        _ => Err(format!("unsupported lazy mode {raw:?}")),
    }
}

fn parse_cache_dtype(raw: &str) -> Result<CacheDtype, String> {
    CacheDtype::parse(raw).ok_or_else(|| format!("unsupported cache dtype {raw:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sizes() {
        assert_eq!(parse_size("1").unwrap(), 1);
        assert_eq!(parse_size("1KiB").unwrap(), 1024);
        assert_eq!(parse_size("2G").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("0.5MiB").unwrap(), 512 * 1024);
        assert!(parse_size("12XB").is_err());
    }

    #[test]
    fn parses_dtypes() {
        assert_eq!(parse_cache_dtype("f16").unwrap(), CacheDtype::F16);
        assert!(parse_cache_dtype("q4").is_err());
    }
}
