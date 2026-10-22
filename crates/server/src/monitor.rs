//! Live system and request monitoring for the status dashboard.

use rustmoe_inference::GenerationResult;
use serde::Serialize;
use std::collections::VecDeque;
use std::process::Command;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::System;

#[derive(Clone, Debug, Serialize)]
pub struct CpuSample {
    pub name: String,
    pub usage_percent: f32,
    pub cores: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemorySample {
    pub total: u64,
    pub used: u64,
    pub available: u64,
    pub usage_percent: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct GpuSample {
    pub index: u32,
    pub name: String,
    pub utilization_percent: f32,
    pub memory_total: u64,
    pub memory_used: u64,
    pub memory_free: u64,
    pub temperature_c: Option<f32>,
    pub power_w: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SystemSample {
    pub timestamp: u64,
    pub cpu: CpuSample,
    pub memory: MemorySample,
    pub gpus: Vec<GpuSample>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ActivitySnapshot {
    pub phase: String,
    pub active: bool,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub context_tokens: u64,
    pub elapsed_ms: f64,
    pub tokens_per_second: f64,
    pub preview: String,
    pub started_unix: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RequestRecord {
    pub started_unix: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub tokens_per_second: f64,
    pub ok: bool,
}

struct Activity {
    phase: String,
    active: bool,
    prompt_tokens: u64,
    generated_tokens: u64,
    context_tokens: u64,
    started: Option<Instant>,
    started_unix: u64,
    preview: String,
    recent: VecDeque<RequestRecord>,
}

pub struct Monitor {
    system: Mutex<System>,
    gpu_cache: Mutex<Option<(Instant, Vec<GpuSample>)>>,
    activity: Mutex<Activity>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn trim_preview(preview: &mut String) {
    if preview.len() <= 360 {
        return;
    }
    let mut cut = preview.len() - 360;
    while cut < preview.len() && !preview.is_char_boundary(cut) {
        cut += 1;
    }
    preview.drain(..cut);
}

impl Default for Monitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Monitor {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_cpu_usage();
        Self {
            system: Mutex::new(system),
            gpu_cache: Mutex::new(None),
            activity: Mutex::new(Activity {
                phase: "idle".into(),
                active: false,
                prompt_tokens: 0,
                generated_tokens: 0,
                context_tokens: 0,
                started: None,
                started_unix: 0,
                preview: String::new(),
                recent: VecDeque::new(),
            }),
        }
    }

    pub fn sample_system(&self) -> SystemSample {
        let mut sys = lock(&self.system);
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        let usage = sys.global_cpu_usage();
        let total = sys.total_memory();
        let available = sys.available_memory();
        let used = total.saturating_sub(available);
        let gpus = self.sample_gpus();
        SystemSample {
            timestamp: now_unix(),
            cpu: CpuSample {
                name: sys
                    .cpus()
                    .first()
                    .map(|c| c.brand().trim().to_string())
                    .unwrap_or_default(),
                usage_percent: usage,
                cores: sys.cpus().len(),
            },
            memory: MemorySample {
                total,
                used,
                available,
                usage_percent: if total == 0 {
                    0.0
                } else {
                    used as f32 / total as f32 * 100.0
                },
            },
            gpus,
        }
    }

    fn sample_gpus(&self) -> Vec<GpuSample> {
        let mut cache = lock(&self.gpu_cache);
        if let Some((at, gpus)) = cache.as_ref() {
            if at.elapsed() < Duration::from_millis(900) {
                return gpus.clone();
            }
        }
        let gpus = query_gpus();
        *cache = Some((Instant::now(), gpus.clone()));
        gpus
    }

    pub fn start(&self, prompt_tokens: u64, context_before: u64) {
        let mut a = lock(&self.activity);
        a.phase = "prefill".into();
        a.active = true;
        a.prompt_tokens = prompt_tokens;
        a.generated_tokens = 0;
        a.context_tokens = context_before.saturating_add(prompt_tokens);
        a.started = Some(Instant::now());
        a.started_unix = now_unix();
        a.preview.clear();
    }

    pub fn token(&self, text: &str) {
        let mut a = lock(&self.activity);
        a.phase = "decode".into();
        a.active = true;
        a.generated_tokens = a.generated_tokens.saturating_add(1);
        a.context_tokens = a.context_tokens.saturating_add(1);
        a.preview.push_str(text);
        trim_preview(&mut a.preview);
    }

    pub fn finish(&self, result: &GenerationResult) {
        let mut a = lock(&self.activity);
        a.phase = "idle".into();
        a.active = false;
        a.prompt_tokens = result.prompt_token_count;
        a.generated_tokens = result.completion_token_count;
        let seconds = if result.decode_ms > 0.0 {
            result.decode_ms / 1000.0
        } else {
            result.total_ms / 1000.0
        };
        let tps = if seconds > 0.0 {
            result.completion_token_count as f64 / seconds
        } else {
            0.0
        };
        let started_unix = a.started_unix;
        a.recent.push_front(RequestRecord {
            started_unix,
            prompt_tokens: result.prompt_token_count,
            completion_tokens: result.completion_token_count,
            prefill_ms: result.prefill_ms,
            decode_ms: result.decode_ms,
            tokens_per_second: tps,
            ok: true,
        });
        while a.recent.len() > 8 {
            a.recent.pop_back();
        }
        a.started = None;
    }

    pub fn fail(&self) {
        let mut a = lock(&self.activity);
        a.phase = "idle".into();
        a.active = false;
        a.started = None;
        let started_unix = a.started_unix;
        let prompt_tokens = a.prompt_tokens;
        let completion_tokens = a.generated_tokens;
        a.recent.push_front(RequestRecord {
            started_unix,
            prompt_tokens,
            completion_tokens,
            prefill_ms: 0.0,
            decode_ms: 0.0,
            tokens_per_second: 0.0,
            ok: false,
        });
        while a.recent.len() > 8 {
            a.recent.pop_back();
        }
    }

    pub fn activity(&self) -> ActivitySnapshot {
        let a = lock(&self.activity);
        let elapsed = a.started.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
        let tps = if elapsed > 0.0 {
            a.generated_tokens as f64 / elapsed
        } else {
            0.0
        };
        ActivitySnapshot {
            phase: a.phase.clone(),
            active: a.active,
            prompt_tokens: a.prompt_tokens,
            generated_tokens: a.generated_tokens,
            context_tokens: a.context_tokens,
            elapsed_ms: elapsed * 1000.0,
            tokens_per_second: tps,
            preview: a.preview.clone(),
            started_unix: a.started_unix,
        }
    }

    pub fn recent(&self) -> Vec<RequestRecord> {
        lock(&self.activity).recent.iter().cloned().collect()
    }
}

fn query_gpus() -> Vec<GpuSample> {
    let mut cmd = Command::new("nvidia-smi");
    cmd.args([
        "--query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let output = match cmd.output() {
        Ok(output) if output.status.success() => output,
        _ => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();
    for line in text.lines() {
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        if cols.len() < 8 {
            continue;
        }
        let index = cols[0].parse::<u32>().ok();
        let util = cols[2].parse::<f32>().ok();
        let total = cols[3].parse::<f64>().ok().map(|m| (m * 1024.0 * 1024.0) as u64);
        let used = cols[4].parse::<f64>().ok().map(|m| (m * 1024.0 * 1024.0) as u64);
        let free = cols[5].parse::<f64>().ok().map(|m| (m * 1024.0 * 1024.0) as u64);
        let temp = cols[6].parse::<f32>().ok();
        let power = cols[7].parse::<f32>().ok();
        if let (Some(index), Some(util), Some(total), Some(used), Some(free)) =
            (index, util, total, used, free)
        {
            gpus.push(GpuSample {
                index,
                name: cols[1].to_string(),
                utilization_percent: util,
                memory_total: total,
                memory_used: used,
                memory_free: free,
                temperature_c: temp,
                power_w: power,
            });
        }
    }
    gpus
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::trim_preview;

    #[test]
    fn preview_truncation_keeps_utf8_boundary() {
        let mut preview = "中".repeat(200);
        trim_preview(&mut preview);
        assert!(preview.len() <= 360);
        assert!(preview.is_char_boundary(preview.len()));
        assert!(preview.chars().all(|c| c == '中'));
    }
}
