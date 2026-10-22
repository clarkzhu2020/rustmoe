//! VRAM, host RAM, and NVMe budget planning.
//!
//! The planner never hands a fixed number to a backend. It computes a budget
//! from the model footprint, KV plan, workspace, safety reserves, and the
//! actual hardware profile. Failure to close the budget is reported before a
//! single weight is mapped.

pub use rustmoe_attention::format_bytes;
use rustmoe_attention::{CacheDtype, KvPlan};

mod auto;
pub use auto::{auto_llama_plan, AutoDevicePlan};
use rustmoe_model::{DeepSeekV41Config, ModelSummary};
use serde::Serialize;
use std::path::Path;
use std::process::Command;
use thiserror::Error;

pub const MIB: u64 = 1024 * 1024;
pub const GIB: u64 = 1024 * MIB;

#[derive(Clone, Debug, Default, Serialize)]
pub struct HardwareProfile {
    pub cpu_name: String,
    pub total_ram: u64,
    pub available_ram: u64,
    pub disks: Vec<DiskInfo>,
    pub gpus: Vec<GpuInfo>,
    pub selected_gpu: Option<usize>,
    pub detection_notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DiskInfo {
    pub name: String,
    pub mount_point: String,
    pub total: u64,
    pub available: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub total_vram: u64,
    pub free_vram: u64,
}

impl HardwareProfile {
    pub fn detect(_model_path: Option<&Path>) -> Self {
        let mut notes = Vec::new();
        let mut sys = sysinfo::System::new_all();
        sys.refresh_memory();
        let cpu_name = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_default();
        let total_ram = sys.total_memory();
        let available_ram = sys.available_memory();
        if total_ram == 0 {
            notes.push("could not detect host RAM; pass --ram to plan explicitly".into());
        }

        let disks = sysinfo::Disks::new_with_refreshed_list()
            .iter()
            .map(|d| DiskInfo {
                name: d.name().to_string_lossy().to_string(),
                mount_point: d.mount_point().to_string_lossy().to_string(),
                total: d.total_space(),
                available: d.available_space(),
            })
            .collect();

        let mut gpus = detect_nvidia_gpus(&mut notes);
        if let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
            filter_visible_devices(&mut gpus, &visible, &mut notes);
        }
        let selected_gpu = gpus
            .iter()
            .enumerate()
            .max_by_key(|(_, g)| g.free_vram)
            .map(|(i, _)| i);

        Self {
            cpu_name,
            total_ram,
            available_ram,
            disks,
            gpus,
            selected_gpu,
            detection_notes: notes,
        }
    }

    pub fn selected_gpu(&self) -> Option<&GpuInfo> {
        self.selected_gpu.and_then(|i| self.gpus.get(i))
    }

    pub fn total_vram(&self) -> u64 {
        self.selected_gpu().map(|g| g.total_vram).unwrap_or(0)
    }

    pub fn free_vram(&self) -> u64 {
        self.selected_gpu().map(|g| g.free_vram).unwrap_or(0)
    }

    pub fn available_disk_for(&self, path: Option<&Path>) -> u64 {
        if let Some(path) = path {
            if let Some(d) = self
                .disks
                .iter()
                .filter(|d| path.starts_with(&d.mount_point))
                .max_by_key(|d| d.mount_point.len())
            {
                return d.available;
            }
        }
        self.disks.iter().map(|d| d.available).max().unwrap_or(0)
    }
}

fn filter_visible_devices(gpus: &mut Vec<GpuInfo>, raw: &str, notes: &mut Vec<String>) {
    let raw = raw.trim();
    if raw.is_empty() || raw == "-1" {
        notes.push("CUDA_VISIBLE_DEVICES hides all GPUs".into());
        gpus.clear();
        return;
    }
    let tokens: Vec<&str> = raw.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    let mut indices = Vec::new();
    for token in tokens {
        match token.parse::<usize>() {
            Ok(index) => indices.push(index),
            Err(_) => {
                notes.push(format!(
                    "CUDA_VISIBLE_DEVICES entry {token} is not a numeric index; GPU selection is left unfiltered"
                ));
                return;
            }
        }
    }
    let mut visible = Vec::new();
    for index in indices {
        match gpus.get(index) {
            Some(gpu) => visible.push(gpu.clone()),
            None => notes.push(format!(
                "CUDA_VISIBLE_DEVICES index {index} does not exist; ignored"
            )),
        }
    }
    if visible.is_empty() {
        notes.push("CUDA_VISIBLE_DEVICES selected no existing GPU".into());
        gpus.clear();
    } else {
        notes.push(format!(
            "CUDA_VISIBLE_DEVICES limits selection to {} GPU(s)",
            visible.len()
        ));
        *gpus = visible;
    }
}

fn detect_nvidia_gpus(notes: &mut Vec<String>) -> Vec<GpuInfo> {
    let command = "nvidia-smi";
    let mut cmd = Command::new(command);
    cmd.args([
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            notes.push(format!("nvidia-smi unavailable: {e}"));
            return Vec::new();
        }
    };
    if !output.status.success() {
        notes.push("nvidia-smi returned a non-zero exit status".into());
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();
    for line in text.lines() {
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        if cols.len() != 3 {
            continue;
        }
        let total = cols[1].parse::<u64>().ok().map(|m| m * MIB);
        let free = cols[2].parse::<u64>().ok().map(|m| m * MIB);
        if let (Some(total_vram), Some(free_vram)) = (total, free) {
            gpus.push(GpuInfo {
                name: cols[0].to_string(),
                total_vram,
                free_vram,
            });
        }
    }
    if gpus.is_empty() {
        notes.push("nvidia-smi produced no parseable GPU rows".into());
    }
    gpus
}

#[derive(Clone, Debug, Serialize)]
pub struct BudgetRequest {
    pub context: u64,
    pub cache_dtype: CacheDtype,
    pub prefill_tokens: u64,
    pub vram_total: Option<u64>,
    pub vram_free: Option<u64>,
    pub ram_total: Option<u64>,
    pub ram_available: Option<u64>,
    pub disk_available: Option<u64>,
    pub gpu_cache_bytes: Option<u64>,
    pub host_cache_bytes: Option<u64>,
    pub display_reserve: u64,
    pub runtime_reserve: u64,
    pub safety_reserve: u64,
    pub ram_reserve: u64,
    pub io_buffer_bytes: u64,
    pub workspace_bytes: Option<u64>,
    pub host_spine: Option<bool>,
    pub nvme_bandwidth: u64,
    pub pcie_bandwidth: u64,
}

impl Default for BudgetRequest {
    fn default() -> Self {
        Self {
            context: 8192,
            cache_dtype: CacheDtype::F16,
            prefill_tokens: 2048,
            vram_total: None,
            vram_free: None,
            ram_total: None,
            ram_available: None,
            disk_available: None,
            gpu_cache_bytes: None,
            host_cache_bytes: None,
            display_reserve: 512 * MIB,
            runtime_reserve: 768 * MIB,
            safety_reserve: 512 * MIB,
            ram_reserve: 8 * GIB,
            io_buffer_bytes: 512 * MIB,
            workspace_bytes: None,
            host_spine: None,
            nvme_bandwidth: 7_000_000_000,
            pcie_bandwidth: 24_000_000_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlanInput {
    pub model_name: String,
    pub context_limit: u64,
    pub context: u64,
    pub block_count: u64,
    pub hidden_size: u64,
    pub vocab_size: u64,
    pub experts_per_token: u64,
    pub summary: ModelSummary,
    pub kv_plan: KvPlan,
}

impl PlanInput {
    pub fn from_model(
        config: &DeepSeekV41Config,
        summary: &ModelSummary,
        context: u64,
        dtype: CacheDtype,
    ) -> Result<Self, MemoryError> {
        if context == 0 {
            return Err(MemoryError::InvalidContext(context));
        }
        if context > config.context_length {
            return Err(MemoryError::ContextTooLong {
                requested: context,
                limit: config.context_length,
            });
        }
        if summary.tensor_count == 0 {
            return Err(MemoryError::NoCatalog);
        }
        Ok(Self {
            model_name: config.name.clone().unwrap_or_else(|| "unnamed".into()),
            context_limit: config.context_length,
            context,
            block_count: config.block_count as u64,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            experts_per_token: config.experts.used_count as u64 * config.block_count as u64,
            summary: summary.clone(),
            kv_plan: KvPlan::estimate(config, context, dtype),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpinePlacement {
    Vram,
    Ram,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Compatible,
    LowSpeed,
    Insufficient,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, Serialize)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub hint: Option<String>,
}

impl Diagnostic {
    fn new(severity: Severity, code: &str, message: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            severity,
            code: code.into(),
            message: message.into(),
            hint: Some(hint.into()),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VramPlan {
    pub total: u64,
    pub free_reported: Option<u64>,
    pub display_reserve: u64,
    pub runtime_reserve: u64,
    pub safety_reserve: u64,
    pub usable: u64,
    pub spine: u64,
    pub kv: u64,
    pub workspace: u64,
    pub expert_cache: u64,
    pub headroom: u64,
    pub spine_placement: SpinePlacement,
}

#[derive(Clone, Debug, Serialize)]
pub struct RamPlan {
    pub total: u64,
    pub available: u64,
    pub reserve: u64,
    pub io_buffers: u64,
    pub resident_spine: u64,
    pub host_cache: u64,
    pub headroom: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DiskPlan {
    pub available: u64,
    pub model_tensor_bytes: u64,
    pub ple_table_bytes: u64,
    pub full_expert_store: u64,
    pub cold_store_is_complete: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExpertIoEstimate {
    pub avg_expert_bytes: u64,
    pub experts_per_token: u64,
    pub cold_expert_bytes_per_token: u64,
    pub gpu_cache_experts: u64,
    pub host_cache_experts: u64,
    pub gpu_cache_fraction: f64,
    pub host_cache_fraction: f64,
    pub cold_disk_ms_per_token: f64,
    pub cold_pcie_ms_per_token: f64,
    pub cold_io_bound_ms_per_token: f64,
    pub nvme_bandwidth: u64,
    pub pcie_bandwidth: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemoryPlan {
    pub status: PlanStatus,
    pub model: String,
    pub context: u64,
    pub context_limit: u64,
    pub hardware: HardwareProfile,
    pub vram: VramPlan,
    pub ram: RamPlan,
    pub disk: DiskPlan,
    pub kv: KvPlan,
    pub expert_io: ExpertIoEstimate,
    pub prefill_chunk: u64,
    pub prefill_activation_bytes_per_token: u64,
    pub diagnostics: Vec<Diagnostic>,
    pub caveats: Vec<String>,
}

impl MemoryPlan {
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics.iter().filter(|d| d.severity == Severity::Error)
    }

    pub fn warnings(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
    }

    pub fn summary_line(&self) -> String {
        format!(
            "{}: ctx={} VRAM usable={} spine={} ({:?}) kv={} gpu-cache={} host-cache={} cold-token={} prefill-chunk={}",
            match self.status {
                PlanStatus::Compatible => "compatible",
                PlanStatus::LowSpeed => "low-speed compatible",
                PlanStatus::Insufficient => "insufficient",
            },
            self.context,
            format_bytes(self.vram.usable),
            format_bytes(self.vram.spine),
            self.vram.spine_placement,
            format_bytes(self.vram.kv),
            format_bytes(self.vram.expert_cache),
            format_bytes(self.ram.host_cache),
            format_bytes(self.expert_io.cold_expert_bytes_per_token),
            self.prefill_chunk
        )
    }
}

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("context length {0} is invalid")]
    InvalidContext(u64),
    #[error("requested context {requested} exceeds model limit {limit}")]
    ContextTooLong { requested: u64, limit: u64 },
    #[error("no tensor catalog is available; load all model shards before planning")]
    NoCatalog,
    #[error("prefill chunk request {0} is invalid")]
    InvalidPrefill(u64),
    #[error("budget arithmetic overflow")]
    Overflow,
    #[error("no CUDA GPU was detected")]
    NoGpu,
    #[error("free VRAM {free} is below the minimum {need}")]
    NotEnoughVram { free: u64, need: u64 },
}

pub fn plan(
    input: &PlanInput,
    hardware: &HardwareProfile,
    request: &BudgetRequest,
) -> Result<MemoryPlan, MemoryError> {
    if request.prefill_tokens > input.context {
        return Err(MemoryError::InvalidPrefill(request.prefill_tokens));
    }
    let mut diagnostics = Vec::new();

    let vram_total = request.vram_total.unwrap_or_else(|| hardware.total_vram());
    let vram_free = request.vram_free.or_else(|| {
        if hardware.gpus.is_empty() {
            None
        } else {
            Some(hardware.free_vram())
        }
    });
    let usable = vram_total
        .saturating_sub(request.display_reserve)
        .saturating_sub(request.runtime_reserve)
        .saturating_sub(request.safety_reserve);
    let kv_bytes = input.kv_plan.total_bytes;
    let activation_per_token = workspace_per_token(input);
    let workspace = request
        .workspace_bytes
        .unwrap_or_else(|| auto_workspace(request.prefill_tokens, activation_per_token));
    let spine = input.summary.dense_spine_bytes;

    let fit_with_spine = spine
        .checked_add(kv_bytes)
        .and_then(|v| v.checked_add(workspace))
        .map(|v| v <= usable)
        .unwrap_or(false);
    let spine_placement = match request.host_spine {
        Some(true) => {
            if spine > 0 {
                SpinePlacement::Ram
            } else {
                SpinePlacement::None
            }
        }
        Some(false) => {
            if !fit_with_spine {
                diagnostics.push(Diagnostic::new(
                    Severity::Error,
                    "vram.spine_does_not_fit",
                    format!(
                        "resident spine {} + KV {} + workspace {} exceeds usable VRAM {}",
                        format_bytes(spine),
                        format_bytes(kv_bytes),
                        format_bytes(workspace),
                        format_bytes(usable)
                    ),
                    "allow spine offload, lower context, or use a larger GPU",
                ));
            }
            SpinePlacement::Vram
        }
        None => {
            if fit_with_spine {
                SpinePlacement::Vram
            } else if spine > 0 {
                diagnostics.push(Diagnostic::new(
                    Severity::Warning,
                    "vram.spine_offloaded",
                    "dense spine does not fit in VRAM; placing it in host memory",
                    "decode will be PCIe-bound; reduce context or use a larger GPU",
                ));
                SpinePlacement::Ram
            } else {
                SpinePlacement::None
            }
        }
    };
    let spine_in_vram = if spine_placement == SpinePlacement::Vram {
        spine
    } else {
        0
    };
    let spine_in_ram = if spine_placement == SpinePlacement::Ram {
        spine
    } else {
        0
    };

    if vram_total == 0 {
        diagnostics.push(Diagnostic::new(
            Severity::Error,
            "vram.unavailable",
            "no CUDA GPU was detected and no --vram override was supplied",
            "this build schedules a single CUDA device; pass --vram after verifying the device",
        ));
    } else if usable < 2 * GIB {
        diagnostics.push(Diagnostic::new(
            Severity::Error,
            "vram.too_small",
            format!("only {} usable VRAM remains after reserves", format_bytes(usable)),
            "at least 2 GiB is required for KV state and CUDA workspace",
        ));
    }
    if let Some(free) = vram_free {
        let required = spine_in_vram
            .saturating_add(kv_bytes)
            .saturating_add(workspace)
            .saturating_add(request.safety_reserve);
        if free < required {
            diagnostics.push(Diagnostic::new(
                Severity::Error,
                "vram.free_too_small",
                format!(
                    "reported free VRAM {} is below the {} required now",
                    format_bytes(free),
                    format_bytes(required)
                ),
                "close other GPU processes or lower the requested context",
            ));
        }
    }

    let fixed = spine_in_vram.saturating_add(kv_bytes).saturating_add(workspace);
    let expert_budget = usable.saturating_sub(fixed);
    if fixed > usable {
        diagnostics.push(Diagnostic::new(
            Severity::Error,
            "vram.fixed_exceeds_usable",
            "KV and workspace alone exceed usable VRAM",
            "lower context, lower prefill batch, or increase --vram",
        ));
    }
    let expert_cache = request
        .gpu_cache_bytes
        .unwrap_or(expert_budget)
        .min(expert_budget);
    if let Some(requested) = request.gpu_cache_bytes {
        if requested > expert_budget {
            diagnostics.push(Diagnostic::new(
                Severity::Warning,
                "vram.cache_clamped",
                format!(
                    "requested GPU expert cache {} was clamped to {}",
                    format_bytes(requested),
                    format_bytes(expert_cache)
                ),
                "the clamp prevents an OOM; adjust context or reserves to cache more",
            ));
        }
    }
    if expert_cache == 0 {
        diagnostics.push(Diagnostic::new(
            Severity::Warning,
            "vram.no_expert_cache",
            "no VRAM remains for a hot expert cache",
            "every routed expert will cross PCIe for each token",
        ));
    }
    let vram_headroom = usable.saturating_sub(fixed.saturating_add(expert_cache));

    let ram_total = request.ram_total.unwrap_or(hardware.total_ram);
    let ram_available = request.ram_available.unwrap_or(hardware.available_ram);
    let ram_reserve = request.ram_reserve;
    if ram_available < ram_reserve {
        diagnostics.push(Diagnostic::new(
            Severity::Error,
            "ram.below_reserve",
            format!(
                "available RAM {} is below the {} reserve",
                format_bytes(ram_available),
                format_bytes(ram_reserve)
            ),
            "close applications or lower --ram-reserve",
        ));
    }
    let host_base = ram_available
        .saturating_sub(ram_reserve)
        .saturating_sub(request.io_buffer_bytes)
        .saturating_sub(spine_in_ram);
    let host_cache = request
        .host_cache_bytes
        .unwrap_or(host_base.saturating_mul(3) / 4)
        .min(host_base)
        .min(input.summary.routed_expert_bytes);
    if host_cache == 0 && input.summary.routed_expert_bytes > 0 {
        diagnostics.push(Diagnostic::new(
            Severity::Warning,
            "ram.no_host_cache",
            "no host expert cache is available",
            "cold experts will be read from NVMe on every miss",
        ));
    }
    if let Some(requested) = request.host_cache_bytes {
        if requested > host_base {
            diagnostics.push(Diagnostic::new(
                Severity::Warning,
                "ram.cache_clamped",
                format!(
                    "requested host cache {} was clamped to {}",
                    format_bytes(requested),
                    format_bytes(host_cache)
                ),
                "keep RAM headroom for the OS and I/O buffers",
            ));
        }
    }
    let ram_headroom = host_base.saturating_sub(host_cache);

    let disk_available = request
        .disk_available
        .unwrap_or_else(|| hardware.available_disk_for(None));
    let disk = DiskPlan {
        available: disk_available,
        model_tensor_bytes: input.summary.total_tensor_bytes,
        ple_table_bytes: input.summary.ple_table_bytes,
        full_expert_store: input.summary.routed_expert_bytes,
        cold_store_is_complete: input.summary.routed_expert_bytes > 0,
    };
    if input.summary.total_tensor_bytes > 0 && disk_available == 0 {
        diagnostics.push(Diagnostic::new(
            Severity::Warning,
            "disk.unknown",
            "free disk space could not be determined",
            "pass --disk to verify the cold store",
        ));
    }

    let avg_expert = input.summary.avg_expert_bytes.max(1);
    let cold_bytes = avg_expert.saturating_mul(input.experts_per_token);
    let gpu_cache_experts = expert_cache / avg_expert;
    let host_cache_experts = host_cache / avg_expert;
    let total_experts = input
        .summary
        .routed_expert_bytes
        .max(1)
        .saturating_div(avg_expert);
    let gpu_fraction = (gpu_cache_experts as f64 / total_experts as f64).min(1.0);
    let host_fraction = (host_cache_experts as f64 / total_experts as f64).min(1.0);
    let cold_disk_ms = cold_bytes as f64 / request.nvme_bandwidth.max(1) as f64 * 1000.0;
    let cold_pcie_ms = cold_bytes as f64 / request.pcie_bandwidth.max(1) as f64 * 1000.0;
    let expert_io = ExpertIoEstimate {
        avg_expert_bytes: avg_expert,
        experts_per_token: input.experts_per_token,
        cold_expert_bytes_per_token: cold_bytes,
        gpu_cache_experts,
        host_cache_experts,
        gpu_cache_fraction: gpu_fraction,
        host_cache_fraction: host_fraction,
        cold_disk_ms_per_token: cold_disk_ms,
        cold_pcie_ms_per_token: cold_pcie_ms,
        cold_io_bound_ms_per_token: cold_disk_ms.max(cold_pcie_ms),
        nvme_bandwidth: request.nvme_bandwidth,
        pcie_bandwidth: request.pcie_bandwidth,
    };
    if gpu_fraction < 0.02 && input.summary.routed_expert_bytes > 0 {
        diagnostics.push(Diagnostic::new(
            Severity::Warning,
            "performance.tiny_gpu_cache",
            format!(
                "GPU cache holds {:.2}% of routed expert weights",
                gpu_fraction * 100.0
            ),
            "expect low-speed compatibility, not interactive decode",
        ));
    }

    let max_by_workspace = (workspace / activation_per_token.max(1)).max(1);
    let requested_prefill = if request.prefill_tokens == 0 {
        input.context.min(max_by_workspace)
    } else {
        request.prefill_tokens
    };
    let prefill_chunk = requested_prefill.min(max_by_workspace).min(input.context);
    if prefill_chunk < requested_prefill {
        diagnostics.push(Diagnostic::new(
            Severity::Warning,
            "prefill.chunk_shrunk",
            format!(
                "prefill chunk reduced from {requested_prefill} to {prefill_chunk} tokens by the workspace budget"
            ),
            "RustMoE will process all prompt chunks; input is never silently truncated",
        ));
    }
    if prefill_chunk == 0 {
        diagnostics.push(Diagnostic::new(
            Severity::Error,
            "prefill.no_room",
            "workspace is too small for one prefill token",
            "increase --workspace or --vram",
        ));
    }

    if input.summary.ple_table_bytes > 0 {
        diagnostics.push(Diagnostic::new(
            Severity::Info,
            "placement.ple_on_disk",
            format!(
                "Engram/PLE table {} remains on NVMe and is looked up per n-gram",
                format_bytes(input.summary.ple_table_bytes)
            ),
            "do not force the full Engram table resident on a 32-128 GB workstation",
        ));
    }

    let status = if diagnostics.iter().any(|d| d.severity == Severity::Error) {
        PlanStatus::Insufficient
    } else if diagnostics.iter().any(|d| d.severity == Severity::Warning) || gpu_fraction < 0.02 {
        PlanStatus::LowSpeed
    } else {
        PlanStatus::Compatible
    };

    Ok(MemoryPlan {
        status,
        model: input.model_name.clone(),
        context: input.context,
        context_limit: input.context_limit,
        hardware: hardware.clone(),
        vram: VramPlan {
            total: vram_total,
            free_reported: vram_free,
            display_reserve: request.display_reserve,
            runtime_reserve: request.runtime_reserve,
            safety_reserve: request.safety_reserve,
            usable,
            spine: spine_in_vram,
            kv: kv_bytes,
            workspace,
            expert_cache,
            headroom: vram_headroom,
            spine_placement,
        },
        ram: RamPlan {
            total: ram_total,
            available: ram_available,
            reserve: ram_reserve,
            io_buffers: request.io_buffer_bytes,
            resident_spine: spine_in_ram,
            host_cache,
            headroom: ram_headroom,
        },
        disk,
        kv: input.kv_plan.clone(),
        expert_io,
        prefill_chunk,
        prefill_activation_bytes_per_token: activation_per_token,
        diagnostics,
        caveats: vec![
            "Compatible means the memory budget closes; it is not a throughput guarantee.".into(),
            "KV bytes are a conservative architectural estimate, not a backend allocation trace.".into(),
            "Expert cache hit rates depend on the real router trace and cannot be predicted statically."
                .into(),
        ],
    })
}

fn workspace_per_token(input: &PlanInput) -> u64 {
    // Activations, attention workspace, and a slice for logits. This is a
    // planning constant until a backend reports its exact workspace.
    input.hidden_size.saturating_mul(4 * 4).max(32 * 1024)
}

fn auto_workspace(prefill_tokens: u64, per_token: u64) -> u64 {
    let activation = prefill_tokens.saturating_mul(per_token);
    activation.max(GIB).saturating_add(512 * MIB)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmoe_attention::KvPlan;
    use rustmoe_model::*;

    fn config() -> DeepSeekV41Config {
        DeepSeekV41Config {
            architecture: "deepseek41".into(),
            name: Some("test".into()),
            block_count: 40,
            context_length: 1_048_576,
            hidden_size: 5120,
            vocab_size: 129280,
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
                    source_layers: vec![2, 8, 14, 20, 24, 28, 32, 36],
                    candidate_source_layer: Some(20),
                    candidate_top_k_blocks: Some(2048),
                    candidate_block_size: Some(8),
                },
            },
            experts: ExpertConfig {
                count: 384,
                used_count: 6,
                shared_count: 1,
                intermediate_size: 2304,
                gating_function: "sqrtsoftplus".into(),
                gating_function_id: 4,
                weights_scale: 1.5,
                weights_norm: true,
            },
            compress_ratios: vec![2; 40],
            kv_source_layers: vec![2, 8, 14, 20],
            index_source_layers: vec![2, 8, 14, 20, 24, 28, 32, 36],
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
            tokenizer: TokenizerInfo {
                model: None,
                pre_tokenizer: None,
                vocab_size: 129280,
                bos_token_id: Some(0),
                eos_token_id: Some(1),
                padding_token_id: Some(2),
                add_bos_token: Some(false),
                add_eos_token: Some(false),
                has_chat_template: true,
            },
        }
    }

    fn input(cfg: &DeepSeekV41Config, context: u64) -> PlanInput {
        let summary = ModelSummary {
            tensor_count: 1000,
            total_tensor_bytes: 500 * 1_000_000_000,
            routed_expert_bytes: 288 * 1_000_000_000,
            dense_spine_bytes: 10 * 1_000_000_000,
            ple_table_bytes: 202 * 1_000_000_000,
            expert_count_per_layer: 384,
            avg_expert_bytes: 18_750_000,
            ..ModelSummary::default()
        };
        PlanInput {
            model_name: "test".into(),
            context_limit: cfg.context_length,
            context,
            block_count: 40,
            hidden_size: 5120,
            vocab_size: 129280,
            experts_per_token: 240,
            summary,
            kv_plan: KvPlan::estimate(cfg, context, CacheDtype::F16),
        }
    }

    fn hw() -> HardwareProfile {
        HardwareProfile {
            cpu_name: "test".into(),
            total_ram: 64 * GIB,
            available_ram: 56 * GIB,
            disks: vec![DiskInfo {
                name: "nvme".into(),
                mount_point: "F:\\".into(),
                total: 4_000 * GIB,
                available: 3_000 * GIB,
            }],
            gpus: vec![GpuInfo {
                name: "RTX 4090".into(),
                total_vram: 24 * GIB,
                free_vram: 23 * GIB,
            }],
            selected_gpu: Some(0),
            detection_notes: vec![],
        }
    }

    fn iq1_summary() -> ModelSummary {
        ModelSummary {
            tensor_count: 1043,
            total_tensor_bytes: 232_979_608_475,
            routed_expert_bytes: 115_240_000_000,
            dense_spine_bytes: 5_870_000_000,
            ple_table_bytes: 110_595_290_592,
            expert_count_per_layer: 384,
            avg_expert_bytes: 7_500_000,
            ..ModelSummary::default()
        }
    }

    #[test]
    fn cuda_visible_devices_filters_gpus() {
        let mut gpus = vec![
            GpuInfo {
                name: "RTX 4090".into(),
                total_vram: 24 * GIB,
                free_vram: 21 * GIB,
            },
            GpuInfo {
                name: "RTX 3060".into(),
                total_vram: 12 * GIB,
                free_vram: 11 * GIB,
            },
        ];
        let mut notes = Vec::new();
        filter_visible_devices(&mut gpus, "1", &mut notes);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "RTX 3060");
        filter_visible_devices(&mut gpus, "", &mut notes);
        assert!(gpus.is_empty());
    }

    #[test]
    fn auto_plan_scales_with_free_vram() {
        let cfg = config();
        let summary = iq1_summary();
        let mut counts = Vec::new();
        for (gib, expect_context) in [(12, 4096), (16, 4096), (24, 8192), (32, 16384)] {
            let mut hw = hw();
            hw.gpus[0].total_vram = gib * GIB;
            hw.gpus[0].free_vram = gib * GIB;
            let plan = auto_llama_plan(&cfg, &summary, &hw, None).unwrap();
            assert_eq!(plan.context, expect_context, "{gib} GiB context");
            assert_eq!(plan.n_gpu_layers, 999);
            assert_eq!(plan.cpu_moe_layers as u32 + plan.gpu_expert_layers, 40);
            assert!(plan.moe_cache_mib >= 0);
            counts.push((gib, plan.gpu_expert_layers));
        }
        assert!(counts[0].1 <= counts[1].1);
        assert!(counts[1].1 < counts[2].1);
        assert!(counts[2].1 < counts[3].1);
    }

    #[test]
    fn auto_plan_tracks_host_ram() {
        let cfg = config();
        let summary = iq1_summary();
        let mut hw = hw();
        hw.available_ram = 24 * GIB;
        hw.gpus[0].total_vram = 32 * GIB;
        hw.gpus[0].free_vram = 32 * GIB;
        let low = auto_llama_plan(&cfg, &summary, &hw, None).unwrap();
        assert_eq!(low.context, 4096);
        hw.available_ram = 256 * GIB;
        let high = auto_llama_plan(&cfg, &summary, &hw, None).unwrap();
        assert_eq!(high.context, 16384);
    }

    #[test]
    fn auto_plan_honours_requested_context() {
        let cfg = config();
        let plan = auto_llama_plan(&cfg, &iq1_summary(), &hw(), Some(2048)).unwrap();
        assert_eq!(plan.context, 2048);
    }

    #[test]
    fn resident_spine_fits_24gb() {
        let cfg = config();
        let input = input(&cfg, 8192);
        let req = BudgetRequest {
            prefill_tokens: 2048,
            ..BudgetRequest::default()
        };
        let plan = plan(&input, &hw(), &req).unwrap();
        assert_eq!(plan.vram.spine_placement, SpinePlacement::Vram);
        assert!(plan.vram.expert_cache > 0);
        assert_ne!(plan.status, PlanStatus::Insufficient);
    }

    #[test]
    fn twelve_gb_offloads_spine() {
        let cfg = config();
        let input = input(&cfg, 8192);
        let mut hw = hw();
        hw.gpus[0].total_vram = 12 * GIB;
        hw.gpus[0].free_vram = 11 * GIB;
        let plan = plan(&input, &hw, &BudgetRequest::default()).unwrap();
        assert_eq!(plan.vram.spine_placement, SpinePlacement::Ram);
        assert_eq!(plan.ram.resident_spine, 10 * 1_000_000_000);
    }

    #[test]
    fn prefill_shrinks_without_truncating() {
        let cfg = config();
        let input = input(&cfg, 8192);
        let req = BudgetRequest {
            prefill_tokens: 4096,
            workspace_bytes: Some(100 * MIB),
            ..BudgetRequest::default()
        };
        let plan = plan(&input, &hw(), &req).unwrap();
        assert!(plan.prefill_chunk < 4096);
        assert!(plan.warnings().any(|d| d.code == "prefill.chunk_shrunk"));
    }

    #[test]
    fn rejects_context_beyond_limit() {
        let cfg = config();
        let err =
            PlanInput::from_model(&cfg, &ModelSummary::default(), 2_000_000, CacheDtype::F16).unwrap_err();
        assert!(matches!(err, MemoryError::ContextTooLong { .. }));
    }
}
