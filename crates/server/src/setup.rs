//! HuggingFace model setup wizard.

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const HF_API: &str = "https://huggingface.co/api/models";
const HF_RESOLVE: &str = "https://huggingface.co";

#[derive(Clone, Copy, Debug)]
pub struct Profile {
    pub id: &'static str,
    pub title: &'static str,
    pub repo: &'static str,
    pub folder: &'static str,
    pub quant: &'static str,
    pub size_gb: u64,
    pub context: u64,
    pub min_ram_gb: u64,
    pub min_vram_gb: u64,
    pub verified: bool,
    pub notes: &'static str,
}

pub const PROFILES: &[Profile] = &[
    Profile {
        id: "deepseek-v4.1-flash-iq1-s",
        title: "DeepSeek-V4.1-Flash-1017 IQ1_S",
        repo: "unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
        folder: "UD-IQ1_S",
        quant: "IQ1_S",
        size_gb: 233,
        context: 131072,
        min_ram_gb: 64,
        min_vram_gb: 24,
        verified: true,
        notes: "Current low-VRAM profile. 217 GiB, 3-5 GPU layers on 24 GB, about 4-5 tok/s.",
    },
    Profile {
        id: "deepseek-v4.1-flash-q4-k-xl",
        title: "DeepSeek-V4.1-Flash-1017 Q4_K_XL",
        repo: "unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
        folder: "UD-Q4_K_XL",
        quant: "Q4_K_XL",
        size_gb: 501,
        context: 131072,
        min_ram_gb: 64,
        min_vram_gb: 24,
        verified: false,
        notes: "500.7 GB. Needs a very large NVMe and heavy SSD paging on 64 GB RAM.",
    },
    Profile {
        id: "deepseek-v4.1-flash-q8-k-xl",
        title: "DeepSeek-V4.1-Flash-1017 Q8_K_XL",
        repo: "unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
        folder: "UD-Q8_K_XL",
        quant: "Q8_K_XL",
        size_gb: 509,
        context: 131072,
        min_ram_gb: 128,
        min_vram_gb: 24,
        verified: false,
        notes: "508.4 GB. Only practical with a large RAM/NVMe server.",
    },
];

#[derive(Debug, Subcommand)]
pub enum SetupCommand {
    /// List HuggingFace-backed model profiles and installed models.
    Models(ModelsArgs),
}

#[derive(Debug, Args)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: ModelsCommand,
}

#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    /// List supported profiles, installed models, and the current pointer.
    List {
        /// Also query HuggingFace for recent GGUF repositories matching this text.
        #[arg(long)]
        search: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Add a model with an interactive wizard.
    Add {
        /// Skip the selection wizard with a profile id.
        #[arg(long)]
        profile: Option<String>,
        /// Destination directory. Default: models/<profile-id>
        #[arg(long)]
        dest: Option<PathBuf>,
        /// Do not ask for confirmation.
        #[arg(long)]
        yes: bool,
        /// Only register an already-downloaded directory.
        #[arg(long)]
        no_download: bool,
        /// HuggingFace token (or use HF_TOKEN).
        #[arg(long)]
        hf_token: Option<String>,
        /// Print what would happen without changing anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Remove a model from the local registry.
    Remove {
        /// Installed name or profile id.
        name: String,
        /// Also delete the model directory.
        #[arg(long)]
        delete_files: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Installed {
    id: String,
    profile: String,
    repo: String,
    path: String,
    files: Vec<String>,
    bytes: u64,
    added_unix: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Registry {
    #[serde(default)]
    models: Vec<Installed>,
}

#[derive(Debug, Clone)]
struct HfFile {
    path: String,
    size: u64,
}

struct HfClient {
    client: reqwest::blocking::Client,
    token: Option<String>,
}

impl HfClient {
    fn new(token: Option<String>) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .user_agent(format!("rustmoe-setup/{}", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self { client, token })
    }

    fn get(&self, url: &str) -> reqwest::blocking::RequestBuilder {
        let mut request = self.client.get(url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        request
    }

    fn tree(&self, repo: &str, revision: &str, folder: &str) -> Result<Vec<HfFile>> {
        let folder = folder.trim_matches('/');
        let url = if folder.is_empty() {
            format!("{HF_API}/{repo}/tree/{revision}")
        } else {
            format!("{HF_API}/{repo}/tree/{revision}/{folder}")
        };
        let response = self
            .get(&url)
            .query(&[("recursive", "true")])
            .timeout(Duration::from_secs(60))
            .send()
            .with_context(|| format!("request HuggingFace tree {url}"))?;
        let status = response.status();
        let body: Value = response
            .json()
            .with_context(|| format!("parse HuggingFace tree response ({status})"))?;
        if !status.is_success() {
            bail!("HuggingFace tree {url} returned {status}: {body}");
        }
        let rows = body
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("HuggingFace tree response is not an array"))?;
        let mut files = Vec::new();
        for row in rows {
            if row.get("type").and_then(Value::as_str) != Some("file") {
                continue;
            }
            let Some(path) = row.get("path").and_then(Value::as_str) else {
                continue;
            };
            if !path.ends_with(".gguf") {
                continue;
            }
            let lower = path.to_ascii_lowercase();
            if lower.contains("mmproj") || lower.contains("engram") || lower.contains("dspark") {
                continue;
            }
            let size = row
                .get("size")
                .and_then(Value::as_u64)
                .or_else(|| row.get("lfs").and_then(|v| v.get("size")).and_then(Value::as_u64))
                .unwrap_or(0);
            files.push(HfFile {
                path: path.to_string(),
                size,
            });
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        if files.is_empty() {
            bail!("no GGUF model files found in {repo}/{folder}");
        }
        Ok(files)
    }

    fn search(&self, query: &str) -> Result<Vec<Value>> {
        let response = self
            .get(HF_API)
            .query(&[
                ("search", query),
                ("filter", "gguf"),
                ("sort", "lastModified"),
                ("direction", "-1"),
                ("limit", "30"),
            ])
            .timeout(Duration::from_secs(60))
            .send()
            .with_context(|| "search HuggingFace models")?;
        let status = response.status();
        let body: Value = response.json().context("parse HuggingFace search")?;
        if !status.is_success() {
            bail!("HuggingFace search returned {status}: {body}");
        }
        Ok(body.as_array().cloned().unwrap_or_default())
    }
}

pub fn run(command: SetupCommand) -> Result<()> {
    match command {
        SetupCommand::Models(args) => match args.command {
            ModelsCommand::List { search, json } => list(search, json),
            ModelsCommand::Add {
                profile,
                dest,
                yes,
                no_download,
                hf_token,
                dry_run,
            } => add(profile, dest, yes, no_download, hf_token, dry_run),
            ModelsCommand::Remove { name, delete_files } => remove(&name, delete_files),
        },
    }
}

fn state_dir() -> PathBuf {
    std::env::var_os("RUSTMOE_STATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".models"))
}

fn registry_path() -> PathBuf {
    state_dir().join("setup-models.json")
}

fn pointer_path() -> PathBuf {
    state_dir().join("model-path.txt")
}

fn load_registry() -> Registry {
    let path = registry_path();
    let Ok(text) = fs::read_to_string(&path) else {
        return Registry::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

fn save_registry(registry: &Registry) -> Result<()> {
    let path = registry_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_vec_pretty(registry)?)
        .with_context(|| format!("write {}", path.display()))
}

fn profile(id: &str) -> Option<&'static Profile> {
    PROFILES
        .iter()
        .find(|profile| profile.id.eq_ignore_ascii_case(id) || profile.quant.eq_ignore_ascii_case(id))
}

fn current_pointer() -> Option<String> {
    fs::read_to_string(pointer_path())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn list(search: Option<String>, json_output: bool) -> Result<()> {
    let registry = load_registry();
    let pointer = current_pointer();

    if let Some(query) = search {
        let client = HfClient::new(std::env::var("HF_TOKEN").ok())?;
        let rows = client.search(&query)?;
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "query": query,
                    "models": rows,
                    "profiles": PROFILES.iter().map(profile_json).collect::<Vec<_>>()
                }))?
            );
            return Ok(());
        }
        println!("HuggingFace GGUF models matching {query:?}, newest first:\n");
        println!(
            "{:<62} {:<12} {:>10}  SUPPORTED",
            "REPOSITORY", "MODIFIED", "DOWNLOADS"
        );
        for row in rows {
            let id = row.get("id").and_then(Value::as_str).unwrap_or("?");
            let modified = row
                .get("lastModified")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .chars()
                .take(10)
                .collect::<String>();
            let downloads = row.get("downloads").and_then(Value::as_u64).unwrap_or(0);
            let supported = PROFILES.iter().any(|p| p.repo == id);
            println!(
                "{:<62} {:<12} {:>10}  {}",
                truncate(id, 62),
                modified,
                downloads,
                if supported {
                    "yes"
                } else {
                    "no (profile not in RustMoE)"
                }
            );
        }
        println!("\nOnly RustMoE profiles can be added with --setup models add.");
        return Ok(());
    }

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "profiles": PROFILES.iter().map(profile_json).collect::<Vec<_>>(),
                "installed": registry.models,
                "current": pointer
            }))?
        );
        return Ok(());
    }

    println!("RustMoE model profiles from HuggingFace:\n");
    println!(
        "{:<38} {:<10} {:>8} {:>9}  STATUS",
        "PROFILE", "QUANT", "SIZE", "CONTEXT"
    );
    for profile in PROFILES {
        let installed = registry.models.iter().find(|m| m.profile == profile.id);
        let status = if let Some(model) = installed {
            if pointer
                .as_deref()
                .is_some_and(|p| p == model.path || p.starts_with(&model.path))
            {
                "current"
            } else {
                "installed"
            }
        } else {
            "not installed"
        };
        println!(
            "{:<38} {:<10} {:>6} GB {:>7}  {:<12} {}",
            profile.id,
            profile.quant,
            profile.size_gb,
            format_context(profile.context),
            status,
            if profile.verified { "" } else { "(not verified)" }
        );
        println!("  {}", profile.title);
        println!("  https://huggingface.co/{}", profile.repo);
        println!("  {}", profile.notes);
    }

    if registry.models.is_empty() {
        println!("\nNo managed models yet. Run: rustmoe --setup models add");
    } else {
        println!("\nInstalled:");
        for model in &registry.models {
            println!(
                "  {}  {}  {}{}",
                model.id,
                format_bytes(model.bytes),
                model.path,
                if pointer
                    .as_deref()
                    .is_some_and(|p| p == model.path || p.starts_with(&model.path))
                {
                    "  [current]"
                } else {
                    ""
                }
            );
        }
    }
    if let Some(pointer) = pointer {
        println!("\nstartdeepseek pointer: {pointer}");
    }
    println!("\nSearch HuggingFace: rustmoe --setup models list --search deepseek");
    Ok(())
}

fn profile_json(profile: &Profile) -> Value {
    json!({
        "id": profile.id,
        "title": profile.title,
        "repo": profile.repo,
        "folder": profile.folder,
        "quant": profile.quant,
        "size_gb": profile.size_gb,
        "context": profile.context,
        "min_ram_gb": profile.min_ram_gb,
        "min_vram_gb": profile.min_vram_gb,
        "verified": profile.verified,
        "notes": profile.notes,
    })
}

fn add(
    profile_id: Option<String>,
    dest: Option<PathBuf>,
    yes: bool,
    no_download: bool,
    hf_token: Option<String>,
    dry_run: bool,
) -> Result<()> {
    let profile = match profile_id {
        Some(id) => {
            profile(&id).ok_or_else(|| anyhow::anyhow!("unknown profile {id:?}; run --setup models list"))?
        }
        None => choose_profile()?,
    };

    let default_dest = PathBuf::from("models").join(profile.id);
    let dest = match dest {
        Some(path) => path,
        None if io::stdin().is_terminal() && !yes => {
            let answer = prompt(&format!("Destination [{}]: ", default_dest.display()))?;
            if answer.is_empty() {
                default_dest
            } else {
                PathBuf::from(answer)
            }
        }
        None => default_dest,
    };
    let token = hf_token.or_else(|| std::env::var("HF_TOKEN").ok());
    let client = HfClient::new(token.clone())?;
    let files = if no_download {
        local_gguf_files(&dest)?
    } else {
        client.tree(profile.repo, "main", profile.folder)?
    };
    let bytes: u64 = files.iter().map(|f| f.size).sum();

    println!("\nModel:       {}", profile.title);
    println!("Profile:     {}", profile.id);
    println!("Repository:  https://huggingface.co/{}", profile.repo);
    println!("Files:       {} GGUF file(s)", files.len());
    println!("Download:    {}", format_bytes(bytes));
    println!("Destination: {}", dest.display());
    println!("Context:     {}", format_context(profile.context));
    println!(
        "Needs:       >= {} GB VRAM, >= {} GB RAM",
        profile.min_vram_gb, profile.min_ram_gb
    );
    if !profile.verified {
        println!("Note:        this profile is not validated on this project yet.");
    }
    if !no_download {
        if let Some(free) = available_space(&dest) {
            println!("Free space:  {}", format_bytes(free));
            if free < bytes {
                bail!(
                    "not enough disk space: need {}, free {}",
                    format_bytes(bytes),
                    format_bytes(free)
                );
            }
        }
        if !yes && !confirm("Download these files now? [Y/n] ", true)? {
            println!("Cancelled.");
            return Ok(());
        }
    }
    if dry_run {
        println!("\nDry run: no files downloaded, registry unchanged.");
        for file in &files {
            println!("  {}  {}", format_bytes(file.size), file.path);
        }
        return Ok(());
    }

    fs::create_dir_all(&dest).with_context(|| format!("create {}", dest.display()))?;
    let mut downloaded = Vec::new();
    if no_download {
        for file in files {
            let path = dest.join(file.path.rsplit('/').next().unwrap_or(&file.path));
            if !path.exists() {
                bail!("missing local file {}", path.display());
            }
            downloaded.push((path, file.size));
        }
    } else {
        for (index, file) in files.iter().enumerate() {
            println!(
                "\n[{}/{}] {} ({})",
                index + 1,
                files.len(),
                file.path,
                format_bytes(file.size)
            );
            let path = dest.join(file.path.rsplit('/').next().unwrap_or(&file.path));
            download_file(&client, profile.repo, "main", &file.path, &path, file.size)?;
            downloaded.push((path, file.size));
        }
    }

    let first = downloaded
        .iter()
        .map(|(path, _)| path)
        .find(|path| path.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .ok_or_else(|| anyhow::anyhow!("no .gguf file in {}", dest.display()))?;
    let pointer = first.to_string_lossy().to_string();
    if let Some(parent) = pointer_path().parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(pointer_path(), &pointer).with_context(|| format!("write {}", pointer_path().display()))?;

    let mut registry = load_registry();
    registry.models.retain(|m| m.profile != profile.id);
    registry.models.push(Installed {
        id: profile.id.to_string(),
        profile: profile.id.to_string(),
        repo: profile.repo.to_string(),
        path: dest.to_string_lossy().to_string(),
        files: downloaded
            .iter()
            .map(|(p, _)| p.to_string_lossy().to_string())
            .collect(),
        bytes,
        added_unix: now_unix(),
    });
    save_registry(&registry)?;

    println!("\nAdded {}", profile.id);
    println!("Model pointer: {pointer}");
    println!("Start with: startdeepseek.bat");
    Ok(())
}

fn choose_profile() -> Result<&'static Profile> {
    if !io::stdin().is_terminal() {
        bail!("not a terminal; pass --profile <id> (see --setup models list)");
    }
    println!("Choose a model:\n");
    for (index, profile) in PROFILES.iter().enumerate() {
        println!(
            "  {}. {}  [{}]  ~{} GB",
            index + 1,
            profile.title,
            profile.quant,
            profile.size_gb
        );
        if !profile.verified {
            println!("     not verified on this project yet");
        }
    }
    let answer = prompt("Select [1]: ")?;
    let index = if answer.trim().is_empty() {
        1
    } else {
        answer
            .trim()
            .parse::<usize>()
            .context("selection must be a number")?
    };
    PROFILES
        .get(index.saturating_sub(1))
        .ok_or_else(|| anyhow::anyhow!("selection out of range"))
}

fn local_gguf_files(dir: &Path) -> Result<Vec<HfFile>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("gguf") {
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.contains("mmproj") || name.contains("engram") || name.contains("dspark") {
            continue;
        }
        let size = entry.metadata()?.len();
        files.push(HfFile {
            path: name.to_string(),
            size,
        });
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    if files.is_empty() {
        bail!("no .gguf files found in {}", dir.display());
    }
    Ok(files)
}

fn download_file(
    client: &HfClient,
    repo: &str,
    revision: &str,
    remote: &str,
    dest: &Path,
    expected: u64,
) -> Result<()> {
    if dest.exists() {
        let existing = fs::metadata(dest)?.len();
        if expected == 0 || existing == expected {
            println!("  already present, skipping");
            return Ok(());
        }
    }
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let mut offset = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if expected > 0 && offset > expected {
        fs::remove_file(&part).ok();
        offset = 0;
    }

    let url = format!(
        "{HF_RESOLVE}/{repo}/resolve/{revision}/{}",
        remote.replace(' ', "%20")
    );
    let mut request = client.get(&url);
    if offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }
    let mut response = request.send().with_context(|| format!("download {url}"))?;
    let status = response.status();
    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && expected > 0 && offset == expected {
        fs::rename(&part, dest)?;
        println!("  complete");
        return Ok(());
    }
    if !status.is_success() {
        bail!("download {url} returned {status}");
    }
    let append = status == reqwest::StatusCode::PARTIAL_CONTENT;
    if append {
        println!("  resuming at {}", format_bytes(offset));
    } else {
        offset = 0;
    }
    let content_len = response.content_length().unwrap_or(0);
    let total = if content_len > 0 {
        offset + content_len
    } else {
        expected
    };
    let mut file = if append {
        OpenOptions::new().append(true).open(&part)?
    } else {
        File::create(&part)?
    };
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut written = offset;
    let mut last = Instant::now();
    loop {
        let read = response.read(&mut buffer).context("read download stream")?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        written += read as u64;
        if last.elapsed() >= Duration::from_secs(1) {
            let total_text = if total > 0 {
                format_bytes(total)
            } else {
                "?".to_string()
            };
            let percent = if total > 0 {
                written as f64 * 100.0 / total as f64
            } else {
                0.0
            };
            print!(
                "\r  {} / {}  ({:.1}%)",
                format_bytes(written),
                total_text,
                percent
            );
            io::stdout().flush().ok();
            last = Instant::now();
        }
    }
    file.flush()?;
    drop(file);
    if expected > 0 && written < expected {
        bail!(
            "incomplete download: {} of {} bytes; rerun to resume",
            written,
            expected
        );
    }
    if dest.exists() {
        fs::remove_file(dest).ok();
    }
    fs::rename(&part, dest).with_context(|| format!("rename {}", dest.display()))?;
    println!("\r  {} complete", format_bytes(written));
    Ok(())
}

fn remove(name: &str, delete_files: bool) -> Result<()> {
    let mut registry = load_registry();
    let Some(index) = registry
        .models
        .iter()
        .position(|m| m.id.eq_ignore_ascii_case(name) || m.profile.eq_ignore_ascii_case(name))
    else {
        bail!("no managed model named {name:?}");
    };
    let model = registry.models.remove(index);
    if let Some(pointer) = current_pointer() {
        if pointer.starts_with(&model.path) {
            fs::remove_file(pointer_path()).ok();
        }
    }
    save_registry(&registry)?;
    println!("Removed {} from the registry.", model.id);
    if delete_files {
        let path = PathBuf::from(&model.path);
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        if canonical.parent().is_none() || canonical == Path::new("/") || canonical == Path::new("C:\\") {
            bail!("refusing to delete unsafe path {}", canonical.display());
        }
        fs::remove_dir_all(&canonical).with_context(|| format!("delete {}", canonical.display()))?;
        println!("Deleted {}", canonical.display());
    } else {
        println!("Files kept at {}", model.path);
    }
    Ok(())
}

fn prompt(question: &str) -> Result<String> {
    print!("{question}");
    io::stdout().flush().ok();
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    Ok(line.trim().to_string())
}

fn confirm(question: &str, default: bool) -> Result<bool> {
    let answer = prompt(question)?;
    if answer.is_empty() {
        return Ok(default);
    }
    Ok(matches!(answer.to_ascii_lowercase().as_str(), "y" | "yes"))
}

fn available_space(path: &Path) -> Option<u64> {
    let mut probe = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    while !probe.exists() {
        if !probe.pop() {
            break;
        }
    }
    fs2::available_space(probe).ok()
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn format_context(tokens: u64) -> String {
    if tokens >= 1024 && tokens % 1024 == 0 {
        format!("{}K", tokens / 1024)
    } else {
        tokens.to_string()
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        text.chars().take(max.saturating_sub(1)).collect::<String>() + "."
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_unique_and_valid() {
        let mut ids = std::collections::HashSet::new();
        for profile in PROFILES {
            assert!(ids.insert(profile.id));
            assert!(!profile.repo.is_empty());
            assert!(!profile.folder.is_empty());
            assert!(profile.size_gb > 0);
        }
    }

    #[test]
    fn formats_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GiB");
    }

    #[test]
    fn formats_context() {
        assert_eq!(format_context(131072), "128K");
        assert_eq!(format_context(8192), "8K");
        assert_eq!(format_context(700), "700");
    }
}
