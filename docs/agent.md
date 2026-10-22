# RustMoE Dev Agent

The Dev Agent adds a guarded local workspace to RustMoE. It can inspect files,
create directories, write code and docs, and (when explicitly enabled) run
PowerShell commands. It is intentionally a tool loop around the local model, not
a hidden remote service.

## Start

    startdeepseek.bat

The default agent root is:

    %CD%\workspace

Open:

    http://127.0.0.1:8080/agent

To let the agent access a real project directory:

    set RUSTMOE_AGENT_ROOT=F:\projects
    set RUSTMOE_AGENT_ALLOW_SHELL=1
    startdeepseek.bat

--agent-allow-shell is required for build/test/install commands. Without it the
agent can still list, read, write, make directories, glob, and grep.

Direct CLI form:

    target\release\rustmoe.exe serve MODEL.gguf --backend llama ^
      --addr 127.0.0.1:8080 --context 131072 --threads 24 ^
      --agent-root F:\projects --agent-allow-shell

## Security model

- Every path is resolved inside the configured agent root.
- .., absolute paths, root paths, and Windows prefixes are rejected.
- Existing symlinks are canonicalized and must remain inside the root.
- Shell commands run with the server user's OS permissions. Use a dedicated
  workspace and a non-administrator account.
- The HTTP API uses the same --api-key bearer token as chat.
- Browser stop closes the event stream; it does not kill an already-running
  server-side tool step. Treat a run as active until it emits finish/error.
- Review generated code and deployment files before production use. The agent
  can verify builds, but it cannot guarantee security or correctness.
- To expose a whole drive, set RUSTMOE_AGENT_ROOT=C:\ (or another drive). The
  path guard is rooted there, so this is powerful and should only be used on a
  trusted local machine.

## Phases

The system prompt drives a complete delivery:

1. plan - PLAN.md: scope, milestones, acceptance criteria.
2. architecture - ARCHITECTURE.md: components, APIs, data, security, failure modes.
3. code - directories and complete source files.
4. ui - real interface and responsive styling.
5. docs - README, API docs, setup and troubleshooting.
6. deploy - Dockerfile, compose/env templates, CI and deployment notes.
7. verify - run build/tests and fix errors (shell enabled).
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
| finish | end the run with a summary |

## API

    POST /v1/agent/run
    Authorization: Bearer ...
    Content-Type: application/json

    {"task":"...", "workspace":".", "max_steps":24}

Response is SSE:

    event: agent
    data: {"type":"start",...}

    event: agent
    data: {"type":"step","step":1,"phase":"plan",...}

    event: agent
    data: {"type":"tool","tool":"write","ok":true,"output":"wrote ..."}

    event: agent
    data: {"type":"finish","summary":"...","files":[...]}

The web page /agent consumes the same stream and renders the execution log and
written file list.

## Limits

- One model call per step; max_tokens=3072 per action.
- max_steps is 1-64 (default 24).
- Observations are truncated to keep context bounded.
- At 4-5 tok/s, a full multi-file project can take a long time. Start with a
  small milestone and inspect /agent.
- A local IQ1 quant is not a replacement for a frontier coding model. The agent
  framework is real; output quality still depends on the model.
