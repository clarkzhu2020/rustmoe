# RustMoE Dev Agent

The Dev Agent adds a guarded local workspace to RustMoE. It can inspect files,
create directories, write code and docs, load skills, and (when explicitly
enabled) run PowerShell. It is a tool loop, not a hidden remote service.

It can use two models:

| Mode | Model | Agent URL | Speed |
|---|---|---|---|
| Fast | Strata / Qwen 125B | http://127.0.0.1:8081/agent | about 90-140 tok/s |
| Deep | local DeepSeek 748B | http://127.0.0.1:8080/agent | about 4-5 tok/s |

Fast and deep share the same agent root, skills, project format, resume state,
export, and delete behavior. Only the model endpoint changes.

## Fast agent with Strata

A standalone binary uses any OpenAI-compatible upstream while keeping the local
tool loop:

    rustmoe-agent.exe --addr 127.0.0.1:8081 ^
      --upstream http://127.0.0.1:8080 ^
      --model qwen3.8-flash-next-iq3_s ^
      --agent-root workspace --skills-dir skills --agent-allow-shell

engine-fast.bat starts this automatically after Strata:

    engine-fast.bat
    # Strata       http://127.0.0.1:8080
    # Fast agent   http://127.0.0.1:8081/agent

The fast agent does not load model weights and uses no VRAM. It sends the system
and user prompt to Strata /v1/chat/completions and parses the returned JSON
action. It still executes every file/shell/skill action locally.

Optional flags:

    --api-key KEY            require Bearer KEY on agent endpoints
    --upstream-api-key KEY   send Bearer KEY to Strata
    --agent-max-steps N      default 24

## Deep agent

    startdeepseek.bat

or:

    engine-deep.bat

Open:

    http://127.0.0.1:8080/agent

The default agent root is:

    %CD%\workspace

To use a real project directory:

    set RUSTMOE_AGENT_ROOT=F:\projects
    set RUSTMOE_AGENT_ALLOW_SHELL=1
    startdeepseek.bat

--agent-allow-shell is required for build/test/install commands. Without it the
agent can still list, read, write, make directories, glob, and grep.

Direct CLI form:

    target\release\rustmoe.exe serve MODEL.gguf --backend llama ^
      --addr 127.0.0.1:8080 --context 131072 --threads 24 ^
      --agent-root F:\projects --agent-allow-shell ^
      --skills-dir skills

## Projects

Every new run creates its own project directory under the agent root:

    chess-20261022-160000/
    todo-20261022-161500/

The name is derived from task keywords plus a local timestamp. Existing projects
are listed in /agent with file count, size, and modified time.

- **Use** selects a project for the next task.
- **Export** downloads a ZIP of the project (max 512 MB).
- **Delete** removes the project directory after confirmation.
- **New** clears the selection so the next run creates a fresh directory.

Only directories containing .rustmoe-project.json are shown as managed projects;
arbitrary folders under the root are not listed. The fast and deep agents share
this project list.

## Interrupted tasks and resume

Each project stores .rustmoe-agent-state.json:

    {
      "status": "running|interrupted|finished|failed",
      "task": "...",
      "step": 3,
      "max_steps": 24,
      "transcript": "...",
      "summary": "..."
    }

If the server restarts while a task is running, or the run is cancelled or hits
its step limit, the project is shown as interrupted. /agent displays:

    Interrupted task found
    <task> - <project> - step 3/24
    [Continue] [Dismiss]

Continue:

- fills in the persisted task and max steps
- sends resume=true
- starts at the saved step and keeps the transcript
- writes the next files in the same project directory

Dismiss hides the banner for this page load; the project remains in the list and
can be selected later.

Resume API:

    POST /v1/agent/run
    {"task":"", "workspace":"my-project-20261022-160000", "resume":true}

The fast agent can resume a project originally started by DeepSeek, but the next
steps use Qwen and may differ in style or quality.

## Security model

- Every path is resolved inside the configured agent root.
- .., absolute paths, root paths, and Windows prefixes are rejected.
- Existing symlinks are canonicalized and must remain inside the root.
- Shell commands run with the server user's OS permissions. Use a dedicated
  workspace and a non-administrator account.
- The HTTP API uses the same --api-key bearer token when configured.
- Browser Stop closes the event stream; it does not kill an already-running
  server-side tool step. Treat a run as active until finish/error.
- Review generated code and deployment files before production use. The agent
  can verify builds, but it cannot guarantee security or correctness.
- To expose a whole drive, set RUSTMOE_AGENT_ROOT=C:\. The path guard is rooted
  there, so this is powerful and should only be used on a trusted local machine.

## Phases

The system prompt drives a complete delivery:

1. plan - PLAN.md: scope, milestones, acceptance criteria.
2. architecture - ARCHITECTURE.md: components, APIs, data, security.
3. code - directories and complete source files.
4. ui - real interface and responsive styling.
5. docs - README, API docs, setup and troubleshooting.
6. deploy - Dockerfile, compose/env templates, CI, deployment notes.
7. verify - build/tests and fix errors (shell enabled).
8. finish - summary and deliverables.

The model returns one JSON action per step:

    {"phase":"code","thought":"create the API entry point","tool":"write","args":{"path":"src/main.rs","content":"..."}}

Tools:

| Tool | Purpose |
|---|---|
| list | list one directory |
| read | read UTF-8 file (max 1 MB) |
| write | write complete file (max 2 MB) |
| mkdir | create directories |
| delete | delete file/directory under root |
| glob | glob up to 200 paths |
| grep | literal search, up to 100 hits |
| shell | run pwsh with timeout (explicit opt-in) |
| skill | load a SKILL.md body or a helper file |
| finish | end the run with a summary |

## Skills

The agent sees a catalog of installed skills and loads one with:

    {"tool":"skill","args":{"name":"web-research"}}

Helper files:

    {"tool":"skill","args":{"name":"comfyui-video","file":"comfyui.mjs"}}

See docs/skills.md. Skills are instructions plus optional helper scripts; they
do not grant new OS permissions by themselves.

## API

Both agents expose:

    GET    /v1/agent/projects
    DELETE /v1/agent/projects/{name}
    GET    /v1/agent/projects/{name}/export
    GET    /v1/agent/skills
    GET    /v1/agent/skills/{name}
    POST   /v1/agent/run
    POST   /v1/agent/cancel

Ports:

    8081  fast agent
    8080  deep agent

/v1/agent/run body:

    {"task":"...", "workspace":"", "max_steps":24, "resume":false}

Empty workspace creates a new project. Pass an existing project name to continue
it. max_steps is 1-64 (default 24); a new run honors the requested value.

Response is SSE:

    event: agent
    data: {"type":"start",...}

    event: agent
    data: {"type":"thinking","step":1,...}

    event: agent
    data: {"type":"step","step":1,"phase":"plan",...}

    event: agent
    data: {"type":"tool","tool":"write","ok":true,"output":"wrote ..."}

    event: agent
    data: {"type":"finish","summary":"...","files":[...]}

Example:

    curl.exe -N -H "Content-Type: application/json" ^
      --data "{\"task\":\"Create a hello.txt and finish\",\"max_steps\":4}" ^
      http://127.0.0.1:8081/v1/agent/run

The browser page consumes the same stream.

## Limits

- One model call per step; max_tokens=3072 per action.
- max_steps is 1-64 (default 24).
- Observations are truncated to keep context bounded.
- Fast agent steps are limited by Strata; deep steps are limited by 4-5 tok/s.
- A full multi-file project can still take many steps. Start with a small
  milestone and inspect /agent.
- Model quality matters: Qwen 125B and DeepSeek IQ1 are not frontier coding
  models. The framework is real; review the output.
