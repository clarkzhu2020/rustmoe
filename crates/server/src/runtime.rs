use crate::monitor::Monitor;
use anyhow::{anyhow, Context, Result};
use rustmoe_attention::CacheDtype;
use rustmoe_backend::{InferenceBackend, SimulatedBackend};
use rustmoe_inference::{Engine, EngineConfig};
use rustmoe_memory::{
    auto_llama_plan, format_bytes, plan, AutoDevicePlan, BudgetRequest, HardwareProfile, MemoryPlan,
    PlanInput, PlanStatus,
};
use rustmoe_model::{DeepSeekV41Config, ModelDescriptor};
use rustmoe_scheduler::{ExpertScheduler, SchedulerConfig};
use rustmoe_telemetry::Metrics;
use rustmoe_tokenizer::BpeTokenizer;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Simulation,
    LlamaCpp,
}

#[derive(Clone, Debug, Serialize)]
pub struct LlamaRuntimeOptions {
    pub n_gpu_layers: Option<i32>,
    pub n_batch: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub moe_cache_mib: Option<i32>,
    pub load_mode: i32,
    pub lazy_mode: i32,
    pub cpu_moe_layers: Option<i32>,
}

impl Default for LlamaRuntimeOptions {
    fn default() -> Self {
        Self {
            n_gpu_layers: None,
            n_batch: 512,
            n_threads: 16,
            n_threads_batch: 16,
            moe_cache_mib: None,
            load_mode: 1,
            lazy_mode: 2,
            cpu_moe_layers: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeOptions {
    pub context: u64,
    pub gpu_cache_bytes: Option<u64>,
    pub host_cache_bytes: Option<u64>,
    pub prefill_chunk: usize,
    pub seed: u64,
    pub prefetch_layers: usize,
    pub model_id: Option<String>,
    pub cache_dtype: CacheDtype,
    pub backend: BackendKind,
    pub auto_device: bool,
    pub llama: LlamaRuntimeOptions,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            context: 4096,
            gpu_cache_bytes: None,
            host_cache_bytes: None,
            prefill_chunk: 2048,
            seed: 0xD5E1_4A11_1017_2026,
            prefetch_layers: 2,
            model_id: None,
            cache_dtype: CacheDtype::F16,
            backend: BackendKind::Simulation,
            auto_device: true,
            llama: LlamaRuntimeOptions::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeInfo {
    pub model_id: String,
    pub model_path: String,
    pub architecture: String,
    pub backend: String,
    pub simulation: bool,
    pub chat_format: String,
    pub context_limit: u64,
    pub context: u64,
    pub prefill_chunk: usize,
    pub tensor_count: usize,
    pub tensor_bytes: u64,
    pub config: DeepSeekV41Config,
    pub plan: Option<MemoryPlan>,
    pub auto_device: bool,
    pub device_plan: Option<AutoDevicePlan>,
    pub warnings: Vec<String>,
    pub started_unix: u64,
}

pub struct Runtime {
    pub engine: Arc<Mutex<Engine>>,
    pub tokenizer: Arc<BpeTokenizer>,
    pub metrics: Metrics,
    pub monitor: Arc<Monitor>,
    pub agent_cancel: Arc<AtomicBool>,
    pub info: RuntimeInfo,
    pub api_key: Option<Arc<str>>,
    started: Instant,
    chat_counter: AtomicU64,
}

fn make_backend(options: &RuntimeOptions) -> Result<(Box<dyn InferenceBackend>, String, bool)> {
    match options.backend {
        BackendKind::Simulation => Ok((
            Box::new(SimulatedBackend::new(options.seed)),
            "simulation".into(),
            true,
        )),
        BackendKind::LlamaCpp => {
            #[cfg(feature = "llama")]
            {
                let backend =
                    rustmoe_backend_llama::LlamaCppBackend::new(rustmoe_backend_llama::LlamaCppOptions {
                        context_size: options.context as u32,
                        n_gpu_layers: options.llama.n_gpu_layers.unwrap_or(999),
                        n_batch: options.llama.n_batch,
                        n_threads: options.llama.n_threads,
                        n_threads_batch: options.llama.n_threads_batch,
                        moe_cache_mib: options.llama.moe_cache_mib.unwrap_or(1024),
                        load_mode: options.llama.load_mode,
                        lazy_mode: options.llama.lazy_mode,
                        cpu_moe_layers: options.llama.cpu_moe_layers.unwrap_or(-1),
                    });
                Ok((Box::new(backend), "llama.cpp".into(), false))
            }
            #[cfg(not(feature = "llama"))]
            {
                anyhow::bail!(
                    "this binary was built without the llama feature; rebuild with --features llama"
                )
            }
        }
    }
}

impl Runtime {
    pub fn load(path: &Path, options: RuntimeOptions) -> Result<Self> {
        Self::load_internal(path, options, false)
    }

    /// Load one shard for metadata inspection or simulation. Split continuity
    /// is not checked and no weight-backed plan can be produced.
    pub fn load_single_shard(path: &Path, options: RuntimeOptions) -> Result<Self> {
        Self::load_internal(path, options, true)
    }

    fn load_internal(path: &Path, mut options: RuntimeOptions, single_shard: bool) -> Result<Self> {
        let descriptor = if single_shard {
            ModelDescriptor::load_single_shard(path)
        } else {
            ModelDescriptor::load(path)
        }
        .with_context(|| format!("load model {}", path.display()))?;
        let hardware = HardwareProfile::detect(Some(path));
        let mut warnings = Vec::new();
        let mut device_plan = None;

        let full_manual = options.llama.n_gpu_layers.is_some()
            && options.llama.cpu_moe_layers.is_some()
            && options.llama.moe_cache_mib.is_some();
        if options.auto_device && !full_manual && options.backend == BackendKind::LlamaCpp {
            let requested = (options.context > 0).then_some(options.context);
            match auto_llama_plan(&descriptor.config, &descriptor.summary, &hardware, requested) {
                Ok(mut plan) => {
                    options.context = plan.context;
                    if options.llama.n_gpu_layers.is_none() {
                        options.llama.n_gpu_layers = Some(plan.n_gpu_layers);
                    }
                    if options.llama.cpu_moe_layers.is_none() {
                        options.llama.cpu_moe_layers = Some(plan.cpu_moe_layers);
                    } else if let Some(cpu) = options.llama.cpu_moe_layers {
                        plan.cpu_moe_layers = cpu;
                        plan.gpu_expert_layers = (descriptor.config.block_count as i32 - cpu).max(0) as u32;
                    }
                    if options.llama.moe_cache_mib.is_none() {
                        options.llama.moe_cache_mib = Some(plan.moe_cache_mib);
                    } else if let Some(cache) = options.llama.moe_cache_mib {
                        let old = plan.moe_cache_mib;
                        if cache != old {
                            let diff = (old - cache).unsigned_abs() as u64 * 1024 * 1024;
                            if cache < old {
                                plan.predicted_vram_bytes = plan.predicted_vram_bytes.saturating_sub(diff);
                                plan.predicted_headroom = plan.predicted_headroom.saturating_add(diff);
                            } else {
                                plan.predicted_vram_bytes = plan.predicted_vram_bytes.saturating_add(diff);
                                plan.predicted_headroom = plan.predicted_headroom.saturating_sub(diff);
                            }
                            plan.moe_cache_mib = cache;
                            plan.notes.push(format!(
                                "explicit {cache} MiB MoE cache overrides the automatic {old} MiB"
                            ));
                        }
                    }
                    warnings.push(format!(
                        "auto device: {} context, {} GPU expert layers, {} host expert layers, {} MiB cache, {} headroom",
                        plan.context,
                        plan.gpu_expert_layers,
                        plan.cpu_moe_layers,
                        plan.moe_cache_mib,
                        format_bytes(plan.predicted_headroom)
                    ));
                    for note in &plan.notes {
                        warnings.push(format!("auto device: {note}"));
                    }
                    device_plan = Some(plan);
                }
                Err(e) => {
                    return Err(anyhow!(
                        "automatic device placement failed: {e}; pass --no-auto-device and explicit flags to override"
                    ));
                }
            }
        }
        if options.context == 0 {
            options.context = 4096;
        }
        if options.context > descriptor.config.context_length {
            return Err(anyhow!(
                "context {} is outside 1..={}",
                options.context,
                descriptor.config.context_length
            ));
        }
        let descriptor = Arc::new(descriptor);
        let tokenizer = Arc::new(BpeTokenizer::from_gguf(&descriptor.gguf).context("load GGUF tokenizer")?);
        if options.backend == BackendKind::LlamaCpp {
            warnings.push(
                "chat uses basic DeepSeek-V4.1 text encoding; tools, images, and thinking mode are not implemented".into(),
            );
        } else {
            warnings.push(
                "chat formatting uses a transparent simple fallback, not the official DeepSeek-V4.1 Jinja template".into(),
            );
        }
        if single_shard {
            if let Some(split) = descriptor.gguf.files.first().and_then(|f| f.split.as_ref()) {
                if split.count > 1 {
                    warnings.push(format!(
                        "single-shard mode: loaded shard {} of {} without the remaining split files",
                        split.no + 1,
                        split.count
                    ));
                }
            }
        }

        let request = BudgetRequest {
            context: options.context,
            cache_dtype: options.cache_dtype,
            prefill_tokens: (options.prefill_chunk as u64).min(options.context),
            gpu_cache_bytes: options.gpu_cache_bytes,
            host_cache_bytes: options.host_cache_bytes,
            ..BudgetRequest::default()
        };
        let plan = match PlanInput::from_model(
            &descriptor.config,
            &descriptor.summary,
            options.context,
            options.cache_dtype,
        ) {
            Ok(input) => match plan(&input, &hardware, &request) {
                Ok(plan) => {
                    if plan.status == PlanStatus::Insufficient {
                        warnings.push(format!(
                            "memory plan reports insufficient resources: {}",
                            plan.summary_line()
                        ));
                    }
                    Some(plan)
                }
                Err(e) => {
                    warnings.push(format!("memory plan unavailable: {e}"));
                    None
                }
            },
            Err(e) => {
                warnings.push(format!(
                    "memory plan unavailable without a complete tensor catalog: {e}"
                ));
                None
            }
        };

        let scheduler_cfg = match SchedulerConfig::from_model(&descriptor, options.prefetch_layers) {
            Ok(c) => c,
            Err(e) => {
                warnings.push(format!("using synthetic expert geometry for simulation: {e}"));
                SchedulerConfig::synthetic(
                    descriptor.config.block_count,
                    descriptor.config.experts.used_count as usize,
                    descriptor.config.experts.shared_count as usize,
                    descriptor.config.experts.count,
                    18_750_000,
                    options.prefetch_layers,
                )
            }
        };

        // The plan already includes and clamps explicit cache requests. Use its
        // values when present so runtime state cannot exceed the budget.
        let mut gpu_cache = plan
            .as_ref()
            .map(|p| p.vram.expert_cache)
            .or(options.gpu_cache_bytes)
            .unwrap_or(2 * 1024 * 1024 * 1024);
        let mut host_cache = plan
            .as_ref()
            .map(|p| p.ram.host_cache)
            .or(options.host_cache_bytes)
            .unwrap_or(8 * 1024 * 1024 * 1024);
        if gpu_cache == 0 {
            gpu_cache = 64 * 1024 * 1024;
            warnings.push("GPU expert cache raised to 64 MiB for simulation accounting".into());
        }
        if host_cache == 0 {
            host_cache = 256 * 1024 * 1024;
            warnings.push("host expert cache raised to 256 MiB for simulation accounting".into());
        }
        let scheduler =
            ExpertScheduler::new(scheduler_cfg, gpu_cache, host_cache).context("create expert scheduler")?;
        let metrics = Metrics::new();
        let (backend, backend_name, simulation) = make_backend(&options)?;
        if !simulation {
            warnings.push(
                "llama.cpp owns routing, MLA, Engram, and kernels; RustMoE expert-cache counters are inactive in this mode".into(),
            );
        }
        let engine_config = EngineConfig::new(options.context, options.prefill_chunk);
        let engine = Engine::new(
            descriptor.clone(),
            tokenizer.clone(),
            backend,
            scheduler,
            metrics.clone(),
            engine_config,
        )
        .context("create inference engine")?;

        let model_id = options
            .model_id
            .clone()
            .or_else(|| descriptor.config.name.clone())
            .or_else(|| path.file_stem().and_then(|s| s.to_str()).map(str::to_string))
            .unwrap_or_else(|| "rustmoe-simulation".into());
        let started_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let info = RuntimeInfo {
            model_id,
            model_path: path.display().to_string(),
            architecture: descriptor.config.architecture.clone(),
            backend: backend_name,
            simulation,
            chat_format: if simulation {
                "simple-fallback".into()
            } else {
                "deepseek-v41-chat".into()
            },
            context_limit: descriptor.config.context_length,
            context: options.context,
            prefill_chunk: options.prefill_chunk,
            tensor_count: descriptor.summary.tensor_count,
            tensor_bytes: descriptor.summary.total_tensor_bytes,
            config: descriptor.config.clone(),
            plan,
            auto_device: device_plan.is_some(),
            device_plan,
            warnings,
            started_unix,
        };
        Ok(Self {
            engine: Arc::new(Mutex::new(engine)),
            tokenizer,
            metrics,
            monitor: Arc::new(Monitor::new()),
            agent_cancel: Arc::new(AtomicBool::new(false)),
            info,
            api_key: None,
            started: Instant::now(),
            chat_counter: AtomicU64::new(1),
        })
    }

    pub fn with_api_key(mut self, key: Option<String>) -> Self {
        self.api_key = key.map(Arc::from);
        self
    }

    pub fn next_id(&self, prefix: &str) -> String {
        let n = self.chat_counter.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{n}")
    }

    pub fn uptime_secs(&self) -> u64 {
        self.started.elapsed().as_secs()
    }

    pub fn model_path(&self) -> PathBuf {
        PathBuf::from(&self.info.model_path)
    }

    pub fn is_simulation(&self) -> bool {
        self.info.simulation
    }
}
