//! Console progress for the long model-load phase.
//!
//! The spinner is only drawn on a terminal so redirected logs and CI output stay
//! clean. It writes to stderr and leaves stdout for commands and JSON.

use std::io::{stderr, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::runtime::RuntimeInfo;
use rustmoe_attention::format_bytes;

pub struct LoadingProgress {
    label: &'static str,
    started: Instant,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    enabled: bool,
}

impl LoadingProgress {
    pub fn start(label: &'static str) -> Self {
        let enabled = stderr().is_terminal();
        let stop = Arc::new(AtomicBool::new(false));
        let handle = if enabled {
            let stop = Arc::clone(&stop);
            Some(thread::spawn(move || {
                let frames = ["|", "/", "-", "\\"];
                let mut i = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let secs = i / 5;
                    let hint = if secs >= 600 {
                        "  (very slow disk; see move-model-to-ssd.bat)"
                    } else if secs >= 300 {
                        "  (still loading; a cold HDD can take 15-25 min)"
                    } else {
                        ""
                    };
                    eprint!("\r{} {}  {}s{}", frames[i % frames.len()], label, secs, hint);
                    let _ = stderr().flush();
                    i += 1;
                    thread::sleep(Duration::from_millis(200));
                }
            }))
        } else {
            None
        };
        Self {
            label,
            started: Instant::now(),
            stop,
            handle,
            enabled,
        }
    }

    /// Stop the spinner and print a one-line completion message.
    pub fn finish(mut self) -> Duration {
        let elapsed = self.started.elapsed();
        self.stop();
        if self.enabled {
            eprintln!("[ok] {} loaded in {:.1}s", self.label, elapsed.as_secs_f64());
        }
        elapsed
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
            if self.enabled {
                eprint!("\r{:80}\r", "");
                let _ = stderr().flush();
            }
        }
    }
}

pub fn print_ready(addr: std::net::SocketAddr, info: &RuntimeInfo) {
    println!("\n{}", "=".repeat(62));
    println!(" RustMoE is ready");
    println!("{}", "=".repeat(62));
    println!(" URL:       http://{addr}");
    println!(" Health:    http://{addr}/health");
    println!(" Model:     {}", info.model_id);
    println!(" Backend:   {} (simulation={})", info.backend, info.simulation);
    println!(" Context:   {} tokens", info.context);
    if let Some(plan) = &info.device_plan {
        println!(" GPU:       {}", plan.gpu_name);
        println!(
            " Placement: {} GPU expert layers / {} host expert layers / {} MiB MoE cache",
            plan.gpu_expert_layers, plan.cpu_moe_layers, plan.moe_cache_mib
        );
        println!(
            " VRAM:      {} free, {} predicted headroom",
            format_bytes(plan.free_vram),
            format_bytes(plan.predicted_headroom)
        );
    }
    println!(" Chat API:  POST http://{addr}/v1/chat/completions");
    println!(" Verify:    testdeepseek.bat");
    println!("{}", "=".repeat(62));
}

impl Drop for LoadingProgress {
    fn drop(&mut self) {
        self.stop();
    }
}
