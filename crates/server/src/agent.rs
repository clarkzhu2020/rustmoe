//! Local development agent: guarded workspace tools and a model tool loop.

use crate::runtime::Runtime;
use anyhow::{anyhow, Context, Result};
use rustmoe_inference::{GenerationConfig, GenerationEvent};
use rustmoe_tokenizer::{format_chat_deepseek_v41, ChatMessage, ChatRole};
use serde::Serialize;
use serde_json::Value;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;
use wait_timeout::ChildExt;

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub root: PathBuf,
    pub allow_shell: bool,
    pub max_steps: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("workspace"),
            allow_shell: false,
            max_steps: 24,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    Start {
        task: String,
        root: String,
        workspace: String,
        max_steps: usize,
    },
    Thinking {
        step: usize,
        message: String,
    },
    Step {
        step: usize,
        phase: String,
        thought: String,
        tool: String,
    },
    Tool {
        step: usize,
        tool: String,
        ok: bool,
        output: String,
    },
    Finish {
        summary: String,
        steps: usize,
        files: Vec<String>,
    },
    Error {
        message: String,
    },
}

pub fn run_agent(
    runtime: Arc<Runtime>,
    config: AgentConfig,
    workspace: String,
    task: String,
    max_steps: usize,
    cancel: Arc<AtomicBool>,
    tx: UnboundedSender<AgentEvent>,
) {
    let emit = |event: AgentEvent| {
        let _ = tx.send(event);
    };
    match execute(runtime, &config, workspace, task, max_steps, &cancel, &emit) {
        Ok(()) => {}
        Err(error) => emit(AgentEvent::Error {
            message: error.to_string(),
        }),
    }
}

fn execute(
    runtime: Arc<Runtime>,
    config: &AgentConfig,
    workspace: String,
    task: String,
    max_steps: usize,
    cancel: &AtomicBool,
    emit: &dyn Fn(AgentEvent),
) -> Result<()> {
    let root = config
        .root
        .canonicalize()
        .with_context(|| format!("agent root {} does not exist", config.root.display()))?;
    let workspace = if workspace.trim().is_empty() || workspace.trim() == "." {
        root.clone()
    } else {
        safe_path(&root, workspace.trim()).map_err(|e| anyhow!(e))?
    };
    std::fs::create_dir_all(&workspace)
        .with_context(|| format!("create workspace {}", workspace.display()))?;
    let max_steps = max_steps.clamp(1, 64);
    emit(AgentEvent::Start {
        task: task.clone(),
        root: root.display().to_string(),
        workspace: workspace.display().to_string(),
        max_steps,
    });
    let mut history = String::new();
    let mut written = Vec::<String>::new();
    let system = agent_system_prompt(&root, config.allow_shell);
    for step in 1..=max_steps {
        if cancel.load(Ordering::Relaxed) {
            emit(AgentEvent::Error {
                message: "cancelled by user".into(),
            });
            return Ok(());
        }
        let tree = list_tree(&workspace, 2);
        let user = format!(
            "TASK: {task}\n\nWORKSPACE TREE:\n{tree}\n\nTRANSCRIPT:\n{history}\n\nReturn exactly one JSON action. Start with {{ and end with }}. No markdown fence."
        );
        let prompt = format_chat_deepseek_v41(
            &[
                ChatMessage {
                    role: ChatRole::System,
                    content: system.clone(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: user,
                },
            ],
            true,
        );
        emit(AgentEvent::Thinking {
            step,
            message: format!("model call {step}/{max_steps}: planning next action"),
        });
        let cfg = GenerationConfig {
            max_tokens: 3072,
            temperature: 0.0,
            top_p: 0.95,
            ..GenerationConfig::default()
        };
        let engine = runtime.engine.clone();
        let mut streamed = String::new();
        let generated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = engine.blocking_lock();
            guard.generate_with(&prompt, &cfg, |event| {
                match &event {
                    GenerationEvent::Token(t) => streamed.push_str(&t.text),
                    GenerationEvent::Text { text } => streamed.push_str(text),
                    GenerationEvent::Finished { .. } => {}
                }
                !(streamed.contains('}') && parse_action(&streamed).is_ok())
            })
        }))
        .map_err(|_| anyhow!("agent generation panicked"))?
        .map_err(|e| anyhow!("agent generation failed: {e}"))?;
        let raw = if streamed.trim().is_empty() {
            generated.text.trim().to_string()
        } else {
            streamed.trim().to_string()
        };
        let action = match parse_action(&raw) {
            Ok(action) => action,
            Err(error) => {
                history.push_str(&format!(
                    "\nSTEP {step}\nMODEL INVALID: {raw}\nOBSERVATION: {error}\n"
                ));
                emit(AgentEvent::Tool {
                    step,
                    tool: "parse".into(),
                    ok: false,
                    output: format!("{error}\nRAW: {}", raw.chars().take(400).collect::<String>()),
                });
                continue;
            }
        };
        emit(AgentEvent::Step {
            step,
            phase: action.phase.clone(),
            thought: action.thought.clone(),
            tool: action.tool.clone(),
        });
        if action.tool == "finish" {
            let summary = arg_str(&action.args, "summary").unwrap_or_else(|| action.thought.clone());
            let files = list_files(&workspace);
            emit(AgentEvent::Finish {
                summary,
                steps: step,
                files,
            });
            return Ok(());
        }
        if cancel.load(Ordering::Relaxed) {
            emit(AgentEvent::Error {
                message: "cancelled by user".into(),
            });
            return Ok(());
        }
        let outcome = execute_tool(&root, &workspace, config.allow_shell, &action);
        let output = match outcome {
            Ok(output) => output,
            Err(error) => format!("ERROR: {error}"),
        };
        if action.tool == "write" {
            if let Some(path) = arg_str(&action.args, "path") {
                if !written.contains(&path) {
                    written.push(path);
                }
            }
        }
        emit(AgentEvent::Tool {
            step,
            tool: action.tool.clone(),
            ok: !output.starts_with("ERROR:"),
            output: output.chars().take(4000).collect(),
        });
        history.push_str(&format!(
            "\nSTEP {step}\nPHASE: {}\nTHOUGHT: {}\nTOOL: {}\nOBSERVATION: {}\n",
            action.phase,
            action.thought,
            action.tool,
            output.chars().take(3000).collect::<String>()
        ));
        if history.len() > 24_000 {
            let keep = history.len() - 24_000;
            let mut cut = keep;
            while cut < history.len() && !history.is_char_boundary(cut) {
                cut += 1;
            }
            history.drain(..cut);
        }
    }
    emit(AgentEvent::Finish {
        summary: format!("Stopped after {max_steps} steps without an explicit finish action."),
        steps: max_steps,
        files: list_files(&workspace),
    });
    Ok(())
}

fn agent_system_prompt(root: &Path, allow_shell: bool) -> String {
    let shell = if allow_shell {
        "shell is ENABLED. Use it to install/build/test. Never run destructive commands outside the workspace."
    } else {
        "shell is DISABLED. Use list/read/write/mkdir/glob/grep only."
    };
    format!(
        r#"You are RustMoE Dev Agent, a senior full-stack engineer and technical writer.
You work only inside this workspace root: {root}

You deliver a production-oriented project through these phases:
1. plan: write PLAN.md with scope, milestones, acceptance criteria.
2. architecture: write ARCHITECTURE.md with components, data model, APIs, security, failure modes.
3. code: create directories and full source files; never leave TODO placeholders.
4. ui: implement the real user interface and responsive styling.
5. docs: write README.md, API docs, setup and troubleshooting.
6. deploy: write Dockerfile, compose/env templates, CI/deployment notes.
7. verify: run build/tests and fix errors.
8. finish: only after the deliverable is coherent and files are written.

{shell}

Reply with exactly ONE JSON object. The first character must be {{ and the last
character must be }}. No markdown fence, no prose, no explanation outside it.

Examples:
{{"phase":"plan","thought":"write the plan","tool":"write","args":{{"path":"PLAN.md","content":"Plan text"}}}}
{{"phase":"code","thought":"write the entry point","tool":"write","args":{{"path":"src/main.rs","content":"..."}}}}
{{"phase":"verify","thought":"done","tool":"finish","args":{{"summary":"...","deliverables":["PLAN.md"]}}}}

Tools:
- {{"tool":"list","args":{{"path":"."}}}}
- {{"tool":"read","args":{{"path":"src/main.rs"}}}}
- {{"tool":"write","args":{{"path":"src/main.rs","content":"..."}}}}
- {{"tool":"mkdir","args":{{"path":"src"}}}}
- {{"tool":"delete","args":{{"path":"old.txt"}}}}
- {{"tool":"glob","args":{{"pattern":"**/*.rs"}}}}
- {{"tool":"grep","args":{{"query":"TODO","path":"."}}}}
- {{"tool":"shell","args":{{"command":"cargo test","timeout_secs":120}}}}
- {{"tool":"finish","args":{{"summary":"what was delivered","deliverables":["..."]}}}}

Rules:
- Paths are relative to the workspace; no absolute paths or ..
- Write complete files, not snippets. Keep each file under 500 lines if possible.
- Inspect before editing when a file already exists.
- Use shell for verification when enabled.
- Do not claim success until build/tests/run evidence exists.
- The final answer must leave a coherent, runnable, documented project.
"#,
        root = root.display(),
        shell = shell
    )
}

#[derive(Debug)]
struct Action {
    phase: String,
    thought: String,
    tool: String,
    args: Value,
}

fn parse_action(raw: &str) -> Result<Action, String> {
    let start = raw.find('{').ok_or("no JSON object found")?;
    let end = raw.rfind('}').ok_or("no closing JSON object found")?;
    if end <= start {
        return Err("invalid JSON object bounds".into());
    }
    let value: Value = serde_json::from_str(&raw[start..=end]).map_err(|e| format!("invalid JSON: {e}"))?;
    let tool = value
        .get("tool")
        .and_then(Value::as_str)
        .ok_or("missing tool")?
        .to_string();
    let args = value.get("args").cloned().unwrap_or(Value::Null);
    Ok(Action {
        phase: value
            .get("phase")
            .and_then(Value::as_str)
            .unwrap_or("code")
            .to_string(),
        thought: value
            .get("thought")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        tool,
        args,
    })
}

fn display_path(path: &Path) -> String {
    let text = path.display().to_string();
    text.strip_prefix("\\\\?\\").unwrap_or(&text).to_string()
}

fn arg_str(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(Value::as_str).map(str::to_string)
}

fn execute_tool(root: &Path, workspace: &Path, allow_shell: bool, action: &Action) -> Result<String, String> {
    match action.tool.as_str() {
        "list" => {
            let path = resolve_under(
                root,
                workspace,
                arg_str(&action.args, "path").as_deref().unwrap_or("."),
            )?;
            list_dir(&path)
        }
        "read" => {
            let raw = arg_str(&action.args, "path").ok_or("read requires path")?;
            let path = resolve_under(root, workspace, &raw)?;
            let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            if bytes.len() > 1_000_000 {
                return Err("file is larger than 1 MB".into());
            }
            String::from_utf8(bytes).map_err(|_| "file is not UTF-8 text".into())
        }
        "write" => {
            let raw = arg_str(&action.args, "path").ok_or("write requires path")?;
            let path = resolve_under(root, workspace, &raw)?;
            let content = arg_str(&action.args, "content").ok_or("write requires content")?;
            if content.len() > 2_000_000 {
                return Err("content is larger than 2 MB".into());
            }
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
            }
            std::fs::write(&path, content).map_err(|e| format!("write {}: {e}", path.display()))?;
            Ok(format!(
                "wrote {} bytes to {}",
                path.metadata().map(|m| m.len()).unwrap_or(0),
                display_path(&path)
            ))
        }
        "mkdir" => {
            let raw = arg_str(&action.args, "path").ok_or("mkdir requires path")?;
            let path = resolve_under(root, workspace, &raw)?;
            std::fs::create_dir_all(&path).map_err(|e| format!("mkdir {}: {e}", path.display()))?;
            Ok(format!("created {}", path.display()))
        }
        "delete" => {
            let raw = arg_str(&action.args, "path").ok_or("delete requires path")?;
            let path = resolve_under(root, workspace, &raw)?;
            if path == workspace || path == root {
                return Err("refusing to delete the workspace root".into());
            }
            if path.is_dir() {
                std::fs::remove_dir_all(&path).map_err(|e| format!("delete dir {}: {e}", path.display()))?;
            } else {
                std::fs::remove_file(&path).map_err(|e| format!("delete file {}: {e}", path.display()))?;
            }
            Ok(format!("deleted {}", path.display()))
        }
        "glob" => {
            let pattern = arg_str(&action.args, "pattern").ok_or("glob requires pattern")?;
            if pattern.contains("..") || Path::new(&pattern).is_absolute() {
                return Err("glob pattern must be relative and stay inside the workspace".into());
            }
            let full = workspace.join(&pattern).to_string_lossy().to_string();
            let mut out = Vec::new();
            for entry in glob::glob(&full)
                .map_err(|e| format!("bad glob: {e}"))?
                .flatten()
                .take(200)
            {
                if let Ok(rel) = entry.strip_prefix(workspace) {
                    out.push(rel.display().to_string());
                }
            }
            Ok(out.join("\n"))
        }
        "grep" => {
            let query = arg_str(&action.args, "query").ok_or("grep requires query")?;
            let path = resolve_under(
                root,
                workspace,
                arg_str(&action.args, "path").as_deref().unwrap_or("."),
            )?;
            grep_path(&path, &query)
        }
        "shell" => {
            if !allow_shell {
                return Err("shell is disabled; restart with --agent-allow-shell".into());
            }
            let command = arg_str(&action.args, "command").ok_or("shell requires command")?;
            let timeout = action
                .args
                .get("timeout_secs")
                .and_then(Value::as_u64)
                .unwrap_or(120)
                .min(600);
            run_shell(workspace, &command, timeout)
        }
        other => Err(format!("unknown tool {other:?}")),
    }
}

fn resolve_under(root: &Path, workspace: &Path, raw: &str) -> Result<PathBuf, String> {
    let rel = Path::new(raw);
    for component in rel.components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            return Err(format!("path {raw:?} escapes the workspace"));
        }
    }
    let joined = workspace.join(rel);
    let mut existing = joined.clone();
    let mut suffix = Vec::new();
    while !existing.exists() {
        let name = existing
            .file_name()
            .ok_or("path has no file name")?
            .to_os_string();
        suffix.push(name);
        existing = existing
            .parent()
            .ok_or("path has no existing ancestor")?
            .to_path_buf();
    }
    let mut resolved = existing
        .canonicalize()
        .map_err(|e| format!("canonicalize {}: {e}", existing.display()))?;
    for part in suffix.into_iter().rev() {
        resolved.push(part);
    }
    if !resolved.starts_with(root) {
        return Err(format!("path {raw:?} is outside the agent root"));
    }
    Ok(resolved)
}

fn safe_path(root: &Path, raw: &str) -> Result<PathBuf, String> {
    resolve_under(root, root, raw)
}

fn list_dir(path: &Path) -> Result<String, String> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(path).map_err(|e| format!("list {}: {e}", path.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let meta = entry.metadata().map_err(|e| e.to_string())?;
        let kind = if meta.is_dir() { "dir" } else { "file" };
        entries.push(format!(
            "{kind}\t{}\t{}",
            entry.file_name().to_string_lossy(),
            meta.len()
        ));
    }
    entries.sort();
    Ok(entries.join("\n"))
}

fn list_tree(root: &Path, depth: usize) -> String {
    fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Vec<String>) {
        if depth == 0 || out.len() > 200 {
            return;
        }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if matches!(name.as_str(), ".git" | "node_modules" | "target" | ".models") {
                    continue;
                }
                let path = entry.path();
                if let Ok(rel) = path.strip_prefix(root) {
                    let is_dir = path.is_dir();
                    out.push(format!("{}{}", rel.display(), if is_dir { "/" } else { "" }));
                    if is_dir {
                        walk(root, &path, depth - 1, out);
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, depth, &mut out);
    out.sort();
    out.join("\n")
}

fn list_files(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        if out.len() > 200 {
            return;
        }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name_os = entry.file_name();
                let name = name_os.to_string_lossy();
                if matches!(name.as_ref(), ".git" | "node_modules" | "target" | ".models") {
                    continue;
                }
                let path = entry.path();
                if path.is_dir() {
                    walk(root, &path, out);
                } else if let Ok(rel) = path.strip_prefix(root) {
                    out.push(rel.display().to_string());
                }
            }
        }
    }
    walk(root, root, &mut out);
    out.sort();
    out
}

fn grep_path(path: &Path, query: &str) -> Result<String, String> {
    let mut files = Vec::new();
    if path.is_file() {
        files.push(path.to_path_buf());
    } else {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            if out.len() > 500 {
                return;
            }
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let name_os = entry.file_name();
                    let name = name_os.to_string_lossy();
                    if matches!(name.as_ref(), ".git" | "node_modules" | "target" | ".models") {
                        continue;
                    }
                    let p = entry.path();
                    if p.is_dir() {
                        walk(&p, out);
                    } else {
                        out.push(p);
                    }
                }
            }
        }
        walk(path, &mut files);
    }
    let mut out = Vec::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            if line.contains(query) {
                out.push(format!("{}:{}:{}", file.display(), index + 1, line));
                if out.len() >= 100 {
                    return Ok(out.join("\n"));
                }
            }
        }
    }
    Ok(out.join("\n"))
}

fn run_shell(workspace: &Path, command: &str, timeout_secs: u64) -> Result<String, String> {
    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("pwsh");
        c.args(["-NoProfile", "-NonInteractive", "-Command", command]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-lc", command]);
        c
    };
    let mut child = cmd
        .current_dir(workspace)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn shell: {e}"))?;
    let status = child
        .wait_timeout(Duration::from_secs(timeout_secs))
        .map_err(|e| format!("wait shell: {e}"))?;
    if status.is_none() {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!("shell timed out after {timeout_secs}s"));
    }
    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut s) = child.stdout.take() {
        let _ = s.read_to_string(&mut stdout);
    }
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_string(&mut stderr);
    }
    let mut out = stdout;
    if !stderr.trim().is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&stderr);
    }
    Ok(out.chars().take(16_000).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_action() {
        let action = parse_action(
            r#"here {"phase":"code","thought":"write it","tool":"write","args":{"path":"a.rs","content":"fn main(){}"}}"#,
        )
        .unwrap();
        assert_eq!(action.phase, "code");
        assert_eq!(action.tool, "write");
        assert_eq!(arg_str(&action.args, "path").as_deref(), Some("a.rs"));
    }

    #[test]
    fn rejects_parent_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir(root.join("src")).unwrap();
        assert!(safe_path(&root, "src/main.rs").is_ok());
        assert!(safe_path(&root, "../escape").is_err());
        assert!(safe_path(&root, "src/../../escape").is_err());
    }

    #[test]
    fn writes_and_reads_inside_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let action = Action {
            phase: "code".into(),
            thought: String::new(),
            tool: "write".into(),
            args: serde_json::json!({"path": "src/main.rs", "content": "fn main() {}"}),
        };
        execute_tool(&root, &root, false, &action).unwrap();
        let read = Action {
            phase: "code".into(),
            thought: String::new(),
            tool: "read".into(),
            args: serde_json::json!({"path": "src/main.rs"}),
        };
        let text = execute_tool(&root, &root, false, &read).unwrap();
        assert_eq!(text, "fn main() {}");
    }
}
