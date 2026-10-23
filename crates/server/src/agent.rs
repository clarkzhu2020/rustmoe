//! Local development agent: guarded workspace tools and a model tool loop.

use crate::runtime::Runtime;
use anyhow::{anyhow, Context, Result};
use rustmoe_inference::{GenerationConfig, GenerationEvent};
use rustmoe_tokenizer::{format_chat_deepseek_v41, ChatMessage, ChatRole};
use serde::{Deserialize, Serialize};
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
    pub skills_dir: PathBuf,
    pub allow_shell: bool,
    pub max_steps: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("workspace"),
            skills_dir: PathBuf::from("skills"),
            allow_shell: false,
            max_steps: 24,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    pub path: String,
}

#[derive(Clone, Debug)]
struct Skill {
    info: SkillInfo,
    body: String,
}

fn load_skills(dir: &Path) -> Vec<Skill> {
    fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(PathBuf, String)>) {
        if depth > 5 || out.len() > 100 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name == "node_modules" || name == ".git" || name == "target" {
                continue;
            }
            if path.is_dir() {
                walk(root, &path, depth + 1, out);
            } else if name == "SKILL.md" {
                if let Ok(body) = std::fs::read_to_string(&path) {
                    let rel = path.strip_prefix(root).unwrap_or(&path).display().to_string();
                    out.push((path, body));
                    let _ = rel;
                }
            }
        }
    }
    let root = dir;
    let mut files = Vec::new();
    walk(root, root, 0, &mut files);
    files
        .into_iter()
        .map(|(path, body)| {
            let fallback = path
                .parent()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "skill".into());
            let (name, description) = parse_skill_header(&body, &fallback);
            Skill {
                info: SkillInfo {
                    name,
                    description,
                    path: path.display().to_string(),
                },
                body,
            }
        })
        .collect()
}

fn parse_skill_header(body: &str, fallback: &str) -> (String, String) {
    let mut name = fallback.to_string();
    let mut description = String::new();
    let trimmed = body.trim_start();
    if let Some(rest) = trimmed.strip_prefix("---") {
        if let Some(end) = rest.find("\n---") {
            let front = &rest[..end];
            for line in front.lines() {
                let Some((key, value)) = line.split_once(':') else {
                    continue;
                };
                let value = value.trim().trim_matches('"').to_string();
                match key.trim() {
                    "name" if !value.is_empty() => name = value,
                    "description" if !value.is_empty() => description = value,
                    _ => {}
                }
            }
        }
    }
    if description.is_empty() {
        description = body
            .lines()
            .find(|line| !line.trim().is_empty() && !line.starts_with("---") && !line.starts_with('#'))
            .unwrap_or("")
            .trim()
            .chars()
            .take(180)
            .collect();
    }
    (name, description)
}

pub fn list_skill_infos(dir: &Path) -> Vec<SkillInfo> {
    load_skills(dir).into_iter().map(|s| s.info).collect()
}

pub fn read_skill(dir: &Path, name: &str) -> Result<String, String> {
    let skills = load_skills(dir);
    skill_body(&skills, name)
}

fn skill_catalog(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return "(none installed)".into();
    }
    let mut out = String::new();
    for skill in skills.iter().take(60) {
        out.push_str(&format!("- {}: {}\n", skill.info.name, skill.info.description));
    }
    out
}

fn find_skill<'a>(skills: &'a [Skill], name: &str) -> Result<&'a Skill, String> {
    skills
        .iter()
        .find(|s| s.info.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| {
            format!(
                "unknown skill {name:?}; available: {}",
                skills
                    .iter()
                    .map(|s| s.info.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

fn skill_body(skills: &[Skill], name: &str) -> Result<String, String> {
    let skill = find_skill(skills, name)?;
    let dir = Path::new(&skill.info.path).parent().unwrap_or(Path::new("."));
    let mut refs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten().take(40) {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|n| n.to_str()) != Some("SKILL.md") {
                if let Ok(rel) = path.strip_prefix(dir) {
                    refs.push(rel.display().to_string());
                }
            }
        }
    }
    refs.sort();
    Ok(format!(
        "# Skill: {}\nPath: {}\nHelper files: {}\n\n{}",
        skill.info.name,
        skill.info.path,
        refs.join(", "),
        skill.body
    ))
}

fn skill_file(skills: &[Skill], name: &str, file: &str) -> Result<String, String> {
    let skill = find_skill(skills, name)?;
    let base = Path::new(&skill.info.path)
        .parent()
        .ok_or("skill has no directory")?;
    let rel = Path::new(file);
    for component in rel.components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            return Err("skill file must stay inside the skill directory".into());
        }
    }
    let path = base.join(rel);
    if !path.starts_with(base) {
        return Err("skill file escapes the skill directory".into());
    }
    let bytes = std::fs::read(&path).map_err(|e| format!("read skill file {}: {e}", path.display()))?;
    if bytes.len() > 1_000_000 {
        return Err("skill file is larger than 1 MB".into());
    }
    String::from_utf8(bytes).map_err(|_| "skill file is not UTF-8".into())
}

pub enum AgentModel {
    Local(Arc<Runtime>),
    Http(HttpAgent),
}

pub struct HttpAgent {
    base_url: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::blocking::Client,
}

impl HttpAgent {
    pub fn new(base_url: String, model: String, api_key: Option<String>) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(None)
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            api_key,
            client,
        }
    }

    fn chat(&self, system: &str, user: &str) -> Result<String, String> {
        let mut request = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&serde_json::json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "max_tokens": 3072,
                "temperature": 0.0,
                "stream": false
            }));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .map_err(|e| format!("upstream request failed: {e}"))?;
        let status = response.status();
        let body: Value = response
            .json()
            .map_err(|e| format!("upstream returned invalid JSON ({status}): {e}"))?;
        if !status.is_success() {
            return Err(format!("upstream HTTP {status}: {body}"));
        }
        body.get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("upstream response has no message content: {body}"))
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    Start {
        task: String,
        project: String,
        root: String,
        workspace: String,
        max_steps: usize,
        resumed: bool,
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
    model: AgentModel,
    config: AgentConfig,
    workspace: String,
    task: String,
    max_steps: usize,
    resume: bool,
    cancel: Arc<AtomicBool>,
    tx: UnboundedSender<AgentEvent>,
) {
    let emit = |event: AgentEvent| {
        let _ = tx.send(event);
    };
    match execute(model, &config, workspace, task, max_steps, resume, &cancel, &emit) {
        Ok(()) => {}
        Err(error) => emit(AgentEvent::Error {
            message: error.to_string(),
        }),
    }
}

fn execute(
    model: AgentModel,
    config: &AgentConfig,
    workspace: String,
    task: String,
    max_steps: usize,
    resume: bool,
    cancel: &AtomicBool,
    emit: &dyn Fn(AgentEvent),
) -> Result<()> {
    let root = config
        .root
        .canonicalize()
        .with_context(|| format!("agent root {} does not exist", config.root.display()))?;
    let (workspace, project) = if workspace.trim().is_empty() {
        let (name, path) = create_project(&root, &task).map_err(|e| anyhow!(e))?;
        (path, name)
    } else {
        let path = safe_path(&root, workspace.trim()).map_err(|e| anyhow!(e))?;
        let name = path
            .strip_prefix(&root)
            .ok()
            .map(|p| p.to_string_lossy().to_string())
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| "root".into());
        (path, name)
    };
    std::fs::create_dir_all(&workspace)
        .with_context(|| format!("create workspace {}", workspace.display()))?;
    let mut state = if resume {
        read_state(&workspace).ok_or_else(|| {
            anyhow!("no resumable state was found in this project; use a normal continue instead")
        })?
    } else {
        RunState::default()
    };
    let task = if task.trim().is_empty() {
        state.task.clone()
    } else {
        task
    };
    let start_step = if resume { state.step } else { 0 };
    let requested_max = if resume {
        max_steps.max(state.max_steps)
    } else {
        max_steps
    };
    let max_steps = if resume && start_step >= requested_max {
        (start_step + 24).min(64)
    } else {
        requested_max.max(start_step + 1).clamp(1, 64)
    };
    state.status = "running".into();
    state.task = task.clone();
    state.max_steps = max_steps;
    state.updated_unix = now_unix();
    if !resume {
        state.transcript.clear();
        state.summary.clear();
        state.step = 0;
    }
    write_state(&workspace, &state);
    emit(AgentEvent::Start {
        task: task.clone(),
        project,
        root: root.display().to_string(),
        workspace: workspace.display().to_string(),
        max_steps,
        resumed: resume,
    });
    let mut history = state.transcript.clone();
    let mut written = Vec::<String>::new();
    let skills_dir = config
        .skills_dir
        .canonicalize()
        .unwrap_or_else(|_| config.skills_dir.clone());
    let skills = load_skills(&skills_dir);
    let system = agent_system_prompt(&root, config.allow_shell, &skills);
    for step in (start_step + 1)..=max_steps {
        if cancel.load(Ordering::Relaxed) {
            state.status = "interrupted".into();
            state.step = step.saturating_sub(1);
            state.transcript = history.clone();
            state.summary = "Cancelled by user; resume is available.".into();
            state.updated_unix = now_unix();
            write_state(&workspace, &state);
            emit(AgentEvent::Error {
                message: "cancelled by user".into(),
            });
            return Ok(());
        }
        let tree = list_tree(&workspace, 2);
        let resume_note = if start_step > 0 {
            format!("\nRESUME FROM STEP {start_step}: previous work is in this project. Inspect the tree before editing and continue from the interruption; do not restart the project.\n")
        } else {
            String::new()
        };
        let user = format!(
            "TASK: {task}\n{resume_note}\nWORKSPACE TREE:\n{tree}\n\nTRANSCRIPT:\n{history}\n\nReturn exactly one JSON action. Start with {{ and end with }}. No markdown fence."
        );
        let raw = match generate_step(&model, &system, &user) {
            Ok(raw) => raw,
            Err(error) => {
                state.status = "interrupted".into();
                state.step = step.saturating_sub(1);
                state.transcript = history.clone();
                state.summary = format!("Generation failed: {error}");
                state.updated_unix = now_unix();
                write_state(&workspace, &state);
                return Err(error);
            }
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
                state.status = "running".into();
                state.step = step.saturating_sub(1);
                state.transcript = history.chars().take(24_000).collect();
                state.updated_unix = now_unix();
                write_state(&workspace, &state);
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
            state.status = "finished".into();
            state.step = step;
            state.transcript = history.chars().take(24_000).collect();
            state.summary = summary.clone();
            state.updated_unix = now_unix();
            write_state(&workspace, &state);
            emit(AgentEvent::Finish {
                summary,
                steps: step,
                files,
            });
            return Ok(());
        }
        if cancel.load(Ordering::Relaxed) {
            state.status = "interrupted".into();
            state.step = step.saturating_sub(1);
            state.transcript = history.clone();
            state.summary = "Cancelled by user; resume is available.".into();
            state.updated_unix = now_unix();
            write_state(&workspace, &state);
            emit(AgentEvent::Error {
                message: "cancelled by user".into(),
            });
            return Ok(());
        }
        let outcome = execute_tool(
            &root,
            &workspace,
            config.allow_shell,
            &skills_dir,
            &skills,
            &action,
        );
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
        state.status = "running".into();
        state.step = step;
        state.transcript = history.chars().take(24_000).collect();
        state.updated_unix = now_unix();
        write_state(&workspace, &state);
        if history.len() > 24_000 {
            let keep = history.len() - 24_000;
            let mut cut = keep;
            while cut < history.len() && !history.is_char_boundary(cut) {
                cut += 1;
            }
            history.drain(..cut);
        }
    }
    state.status = "interrupted".into();
    state.step = max_steps;
    state.transcript = history.chars().take(24_000).collect();
    state.summary = format!("Stopped after {max_steps} steps without an explicit finish action.");
    state.updated_unix = now_unix();
    write_state(&workspace, &state);
    emit(AgentEvent::Finish {
        summary: format!("Stopped after {max_steps} steps without an explicit finish action."),
        steps: max_steps,
        files: list_files(&workspace),
    });
    Ok(())
}

fn agent_system_prompt(root: &Path, allow_shell: bool, skills: &[Skill]) -> String {
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

Available skills (load details with the skill tool):
{skills}

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
- {{"tool":"skill","args":{{"name":"web-research"}}}}
- {{"tool":"skill","args":{{"name":"comfyui-video","file":"comfyui.mjs"}}}}
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
        shell = shell,
        skills = skill_catalog(skills)
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

fn execute_tool(
    root: &Path,
    workspace: &Path,
    allow_shell: bool,
    skills_dir: &Path,
    skills: &[Skill],
    action: &Action,
) -> Result<String, String> {
    match action.tool.as_str() {
        "skill" => {
            let name = arg_str(&action.args, "name").ok_or("skill requires name")?;
            match arg_str(&action.args, "file") {
                Some(file) => skill_file(skills, &name, &file),
                None => skill_body(skills, &name),
            }
        }
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
            run_shell(workspace, skills_dir, &command, timeout)
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
                if matches!(
                    name.as_str(),
                    ".git"
                        | "node_modules"
                        | "target"
                        | ".models"
                        | ".rustmoe-project.json"
                        | ".rustmoe-agent-state.json"
                ) {
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
                if matches!(
                    name.as_ref(),
                    ".git"
                        | "node_modules"
                        | "target"
                        | ".models"
                        | ".rustmoe-project.json"
                        | ".rustmoe-agent-state.json"
                ) {
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
                    if matches!(
                        name.as_ref(),
                        ".git"
                            | "node_modules"
                            | "target"
                            | ".models"
                            | ".rustmoe-project.json"
                            | ".rustmoe-agent-state.json"
                    ) {
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

fn run_shell(
    workspace: &Path,
    skills_dir: &Path,
    command: &str,
    timeout_secs: u64,
) -> Result<String, String> {
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
        .env("RUSTMOE_SKILLS_DIR", skills_dir)
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
        execute_tool(&root, &root, false, Path::new("skills"), &[], &action).unwrap();
        let read = Action {
            phase: "code".into(),
            thought: String::new(),
            tool: "read".into(),
            args: serde_json::json!({"path": "src/main.rs"}),
        };
        let text = execute_tool(&root, &root, false, Path::new("skills"), &[], &read).unwrap();
        assert_eq!(text, "fn main() {}");
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectInfo {
    pub name: String,
    pub title: String,
    pub task: String,
    pub created_unix: u64,
    pub updated_unix: u64,
    pub files: usize,
    pub bytes: u64,
    pub status: String,
    pub step: usize,
    pub max_steps: usize,
    pub summary: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunState {
    #[serde(default = "default_status")]
    status: String,
    #[serde(default)]
    task: String,
    #[serde(default)]
    step: usize,
    #[serde(default)]
    max_steps: usize,
    #[serde(default)]
    updated_unix: u64,
    #[serde(default)]
    transcript: String,
    #[serde(default)]
    summary: String,
}

fn default_status() -> String {
    "new".into()
}

impl Default for RunState {
    fn default() -> Self {
        Self {
            status: "new".into(),
            task: String::new(),
            step: 0,
            max_steps: 24,
            updated_unix: now_unix(),
            transcript: String::new(),
            summary: String::new(),
        }
    }
}

fn state_path(workspace: &Path) -> PathBuf {
    workspace.join(".rustmoe-agent-state.json")
}

fn read_state(workspace: &Path) -> Option<RunState> {
    std::fs::read_to_string(state_path(workspace))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

fn write_state(workspace: &Path, state: &RunState) {
    if let Ok(body) = serde_json::to_string_pretty(state) {
        let _ = std::fs::write(state_path(workspace), body);
    }
}

pub fn mark_interrupted_projects(root: &Path) -> Result<usize, String> {
    let root = root
        .canonicalize()
        .map_err(|e| format!("canonicalize root {}: {e}", root.display()))?;
    let mut changed = 0;
    for entry in std::fs::read_dir(&root).map_err(|e| e.to_string())?.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(mut state) = read_state(&path) else {
            continue;
        };
        if state.status == "running" {
            state.status = "interrupted".into();
            state.updated_unix = now_unix();
            state.summary = "Server restarted while this task was running.".into();
            write_state(&path, &state);
            changed += 1;
        }
    }
    Ok(changed)
}

pub fn create_project(root: &Path, task: &str) -> Result<(String, PathBuf), String> {
    let root = root
        .canonicalize()
        .map_err(|e| format!("canonicalize root {}: {e}", root.display()))?;
    let base = project_slug(task);
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    for attempt in 1..1000 {
        let name = if attempt == 1 {
            format!("{base}-{stamp}")
        } else {
            format!("{base}-{stamp}-{attempt}")
        };
        let path = root.join(&name);
        if path.exists() {
            continue;
        }
        std::fs::create_dir_all(&path).map_err(|e| format!("create project {}: {e}", path.display()))?;
        let now = now_unix();
        let meta = serde_json::json!({
            "name": name,
            "title": task.chars().take(80).collect::<String>(),
            "task": task,
            "created_unix": now,
        });
        let _ = std::fs::write(path.join(".rustmoe-project.json"), meta.to_string());
        return Ok((name, path));
    }
    Err("could not allocate a unique project directory".into())
}

pub fn list_projects(root: &Path) -> Result<Vec<ProjectInfo>, String> {
    list_projects_with_active(root, false)
}

pub fn list_projects_with_active(root: &Path, active: bool) -> Result<Vec<ProjectInfo>, String> {
    let root = root
        .canonicalize()
        .map_err(|e| format!("canonicalize root {}: {e}", root.display()))?;
    let mut out = Vec::new();
    let entries = std::fs::read_dir(&root).map_err(|e| format!("list {}: {e}", root.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        let mut files = 0usize;
        let mut bytes = 0u64;
        let mut updated = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        count_files(&path, &mut files, &mut bytes, &mut updated);
        let Some(meta) = std::fs::read_to_string(path.join(".rustmoe-project.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        else {
            continue;
        };
        let title = meta
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or(&name)
            .to_string();
        let task = meta.get("task").and_then(Value::as_str).unwrap_or("").to_string();
        let created = meta
            .get("created_unix")
            .and_then(Value::as_u64)
            .unwrap_or(updated);
        let run = read_state(&path).unwrap_or_default();
        let status = if run.status == "running" && !active {
            "interrupted".to_string()
        } else {
            run.status.clone()
        };
        out.push(ProjectInfo {
            name,
            title,
            task,
            created_unix: created,
            updated_unix: updated.max(run.updated_unix),
            files,
            bytes,
            status,
            step: run.step,
            max_steps: run.max_steps,
            summary: run.summary,
        });
    }
    out.sort_by(|a, b| b.updated_unix.cmp(&a.updated_unix));
    Ok(out)
}

pub fn delete_project(root: &Path, name: &str) -> Result<(), String> {
    let root = root
        .canonicalize()
        .map_err(|e| format!("canonicalize root {}: {e}", root.display()))?;
    let path = safe_path(&root, name)?;
    if path == root {
        return Err("refusing to delete the agent root".into());
    }
    if !path.is_dir() {
        return Err("project is not a directory".into());
    }
    std::fs::remove_dir_all(&path).map_err(|e| format!("delete {}: {e}", path.display()))
}

pub fn export_project(root: &Path, name: &str) -> Result<Vec<u8>, String> {
    let root = root
        .canonicalize()
        .map_err(|e| format!("canonicalize root {}: {e}", root.display()))?;
    let dir = safe_path(&root, name)?;
    if !dir.is_dir() {
        return Err("project is not a directory".into());
    }
    let mut files = Vec::new();
    collect_export_files(&dir, &dir, &mut files, 0)?;
    let mut total = 0u64;
    for (_, path) in &files {
        total += std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    if total > 512 * 1024 * 1024 {
        return Err("project is larger than 512 MB".into());
    }
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (rel, path) in files {
        let data = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        zip.start_file(rel, options)
            .map_err(|e| format!("zip entry: {e}"))?;
        std::io::Write::write_all(&mut zip, &data).map_err(|e| format!("zip write: {e}"))?;
    }
    let cursor = zip.finish().map_err(|e| format!("finish zip: {e}"))?;
    Ok(cursor.into_inner())
}

fn collect_export_files(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(String, PathBuf)>,
    depth: usize,
) -> Result<(), String> {
    if depth > 32 || out.len() > 20000 {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)
        .map_err(|e| format!("list {}: {e}", dir.display()))?
        .flatten()
    {
        let name = entry.file_name().to_string_lossy().to_string();
        if matches!(
            name.as_str(),
            ".git"
                | "node_modules"
                | "target"
                | ".models"
                | ".rustmoe-project.json"
                | ".rustmoe-agent-state.json"
        ) {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            collect_export_files(root, &path, out, depth + 1)?;
        } else if let Ok(rel) = path.strip_prefix(root) {
            out.push((rel.to_string_lossy().replace('\\', "/"), path));
        }
    }
    Ok(())
}

fn count_files(dir: &Path, files: &mut usize, bytes: &mut u64, updated: &mut u64) {
    if *files > 100000 {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Ok(meta) = entry.metadata() {
                if let Ok(modified) = meta.modified() {
                    if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                        *updated = (*updated).max(duration.as_secs());
                    }
                }
                if meta.is_dir() {
                    count_files(&path, files, bytes, updated);
                } else {
                    *files += 1;
                    *bytes += meta.len();
                }
            }
        }
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn project_slug(task: &str) -> String {
    let lower = task.to_lowercase();
    let known: &[(&str, &str)] = &[
        ("象棋", "chess"),
        ("chess", "chess"),
        ("游戏", "game"),
        ("game", "game"),
        ("待办", "todo"),
        ("todo", "todo"),
        ("任务管理", "task-manager"),
        ("博客", "blog"),
        ("blog", "blog"),
        ("商城", "shop"),
        ("shop", "shop"),
        ("聊天", "chat"),
        ("chat", "chat"),
        ("爬虫", "crawler"),
        ("crawler", "crawler"),
        ("计算器", "calculator"),
        ("calculator", "calculator"),
        ("管理", "manager"),
        ("management", "manager"),
        ("登录", "auth"),
        ("auth", "auth"),
        ("文档", "docs"),
        ("docs", "docs"),
        ("api", "api"),
        ("dashboard", "dashboard"),
        ("website", "website"),
        ("网站", "website"),
        ("portfolio", "portfolio"),
    ];
    for (needle, slug) in known {
        if lower.contains(needle) {
            return (*slug).to_string();
        }
    }
    let mut ascii = String::new();
    for word in lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() >= 2)
    {
        if !ascii.is_empty() {
            ascii.push('-');
        }
        ascii.push_str(word);
        if ascii.len() >= 32 {
            break;
        }
    }
    let ascii = ascii.trim_matches('-').to_string();
    if ascii.is_empty() {
        "project".into()
    } else {
        ascii
    }
}

#[cfg(test)]
mod project_tests {
    use super::*;

    #[test]
    fn slug_uses_task_language() {
        assert_eq!(project_slug("开发一个中国象棋游戏"), "chess");
        assert_eq!(project_slug("build a todo app"), "todo");
        assert_eq!(project_slug("random words here"), "random-words-here");
    }

    #[test]
    fn project_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let (name, path) = create_project(&root, "build a todo app").unwrap();
        assert!(name.starts_with("todo-"));
        std::fs::write(path.join("README.md"), "# Todo\n").unwrap();
        let list = list_projects(&root).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, name);
        assert!(list[0].files >= 1);
        let zip = export_project(&root, &name).unwrap();
        assert!(zip.len() > 20);
        delete_project(&root, &name).unwrap();
        assert!(list_projects(&root).unwrap().is_empty());
    }
}

#[cfg(test)]
mod resume_tests {
    use super::*;

    #[test]
    fn marks_running_project_interrupted() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let (_name, path) = create_project(&root, "build a todo app").unwrap();
        let state = RunState {
            status: "running".into(),
            task: "build a todo app".into(),
            step: 3,
            max_steps: 24,
            updated_unix: now_unix(),
            transcript: "STEP 3".into(),
            summary: String::new(),
        };
        write_state(&path, &state);
        mark_interrupted_projects(&root).unwrap();
        let state = read_state(&path).unwrap();
        assert_eq!(state.status, "interrupted");
        assert_eq!(state.step, 3);
        let projects = list_projects(&root).unwrap();
        assert_eq!(projects[0].status, "interrupted");
    }
}

#[cfg(test)]
mod skill_tests {
    use super::*;

    #[test]
    fn loads_installed_skills() {
        let skills = load_skills(Path::new("../../skills"));
        let names: Vec<&str> = skills.iter().map(|s| s.info.name.as_str()).collect();
        assert!(names.contains(&"web-research"), "{names:?}");
        assert!(names.contains(&"comfyui-video"), "{names:?}");
        assert!(names.contains(&"agent-experience"), "{names:?}");
        assert!(skill_body(&skills, "web-research")
            .unwrap()
            .contains("search.mjs"));
    }
}

fn generate_step(model: &AgentModel, system: &str, user: &str) -> Result<String> {
    match model {
        AgentModel::Http(agent) => agent
            .chat(system, user)
            .map(|text| text.trim().to_string())
            .map_err(|e| anyhow!("agent upstream failed: {e}")),
        AgentModel::Local(runtime) => {
            let prompt = format_chat_deepseek_v41(
                &[
                    ChatMessage {
                        role: ChatRole::System,
                        content: system.to_string(),
                    },
                    ChatMessage {
                        role: ChatRole::User,
                        content: user.to_string(),
                    },
                ],
                true,
            );
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
            if streamed.trim().is_empty() {
                Ok(generated.text.trim().to_string())
            } else {
                Ok(streamed.trim().to_string())
            }
        }
    }
}
