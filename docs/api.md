# HTTP API

Base URL example: http://127.0.0.1:8080

Two backends are available. The default simulation is for tests. Real inference
is selected with --backend llama and requires a RustMoE binary built with the
llama feature plus the reference llama.cpp DLLs. Real responses carry
"simulation": false, backend "llama.cpp", and the header x-rustmoe-backend.
Simulation responses carry "simulation": true and x-rustmoe-simulation: true.

## Authentication

Start with --api-key secret to require:

    Authorization: Bearer secret

Health, model listing, and metrics remain public. Inference, tokenization,
runtime state, and cache reset require the key.

## POST /v1/agent/run

Local development-agent loop. Body: task, workspace, max_steps. Returns SSE
events: start, step, tool, finish, error. Requires auth when enabled. The agent
only accesses paths under the server's --agent-root; shell is disabled unless
--agent-allow-shell is set. See docs/agent.md.

## Local agent project endpoints

    GET    /v1/agent/projects
    DELETE /v1/agent/projects/{name}
    GET    /v1/agent/projects/{name}/export

Projects are directories under --agent-root that contain
.rustmoe-project.json. Export returns application/zip. New runs with an empty
workspace create a name from task keywords plus a timestamp. A run can pass
resume=true (workspace is required) to continue from .rustmoe-agent-state.json;
the event stream and project list expose status and step.

## GET /agent

Dev Agent browser UI. It starts /v1/agent/run, streams execution events, and
collects written file paths.

## GET /chat

Browser chat UI. It streams /v1/chat/completions with `stream: true`, keeps
history client-side, and shows /v1/telemetry in the sidebar. Max new tokens is
estimated from the prompt in the range 8192-262144, capped by context; on
finish_reason=length it automatically continues in the same bubble. Auto-sizing
and auto-continue can be switched off; manual Continue remains available.
Sessions (history, title, timestamps) are stored in browser localStorage, so
they survive a reload and can be switched, renamed, or deleted in the sidebar.
The page is public; the API key can be entered in the page and is stored in
localStorage.

## GET /status

Dark live dashboard. The page is public; it polls /health, /v1/models,
/v1/runtime, and /v1/telemetry every second. If --api-key is configured, the
page shows a key field and stores the bearer token in localStorage. It renders
service, context, live VRAM, device plan, backend cache, latency history,
warnings, API access URLs/cURL, and raw JSON.

## GET /v1/telemetry

Live dashboard feed. Requires auth when enabled. Returns CPU/GPU/RAM/VRAM,
current activity, the last eight requests, and metrics:

    {
      "system": {"cpu": {...}, "memory": {...}, "gpus": [...]},
      "activity": {"phase": "decode", "generated_tokens": 8, "tokens_per_second": 4.5},
      "recent": [{"prompt_tokens": 8, "completion_tokens": 8, "tokens_per_second": 5.16}],
      "metrics": {...}
    }

## GET /v1/system

Live hardware snapshot from nvidia-smi and sysinfo. Requires auth when enabled.

    {"hardware":{"cpu_name":"...","available_ram":...,"gpus":[...],"disks":[...]}}

## GET /health

    {"status":"ok","model":"...","backend":"simulation","simulation":true,"uptime_secs":12}

## GET /v1/models

OpenAI list shape, plus top-level context_length, max_context_length,
context_window, and max_model_len fields. A rustmoe object also carries
architecture, backend, simulation, context length, chat format, and tensor
count. The 128K run reports 131072/1048576/131072/131072.

## GET /v1/runtime

Returns runtime configuration, memory plan if a full catalog was loaded,
scheduler statistics, cache snapshots, backend_cache, and metrics. The runtime
object contains auto_device and, when automatic placement ran, device_plan with
the selected GPU, free VRAM, context, GPU/host expert layers, cache size, and
predicted headroom. In real mode backend_cache contains the cumulative llama.cpp
MoE cache hits, misses, uploaded_bytes, copied_hits, and copied_bytes. Requires
auth when enabled.

## GET /metrics

Prometheus text. Stable metric names include:

- rustmoe_requests_total
- rustmoe_requests_active
- rustmoe_errors_total
- rustmoe_prompt_tokens_total
- rustmoe_completion_tokens_total
- rustmoe_decode_steps_total
- rustmoe_cache_gpu_hits_total
- rustmoe_cache_host_hits_total
- rustmoe_cache_disk_misses_total
- rustmoe_cache_hit_rate
- rustmoe_disk_bytes_read_total
- rustmoe_h2d_bytes_total
- rustmoe_cache_evictions_total
- rustmoe_context_tokens
- rustmoe_llama_moe_cache_hits_total (real backend)
- rustmoe_llama_moe_cache_misses_total (real backend)
- rustmoe_llama_moe_cache_hit_rate (real backend)
- rustmoe_llama_moe_cache_uploaded_bytes_total (real backend)
- rustmoe_ttft_ms (histogram)
- rustmoe_prefill_ms (histogram)
- rustmoe_decode_ms (histogram)

## POST /v1/tokenize

Request:

    {"text":"Hello, world!","add_special":false}

Response:

    {"ids":[19923,14,2058,3],"tokens":["Hello",",","Ġworld","!"],
     "decoded":"Hello, world!","pre_tokenizer":"joyai-llm",
     "pre_tokenizer_exact":false}

## POST /v1/chat/completions

Request:

    {
      "model": "RustMoE Tiny DS41 Fixture",
      "messages": [
        {"role": "user", "content": "Hi"}
      ],
      "max_tokens": 8,
      "temperature": 0.0,
      "top_p": 1.0,
      "top_k": 0,
      "seed": 7,
      "stream": false
    }

Supported roles: system, user, assistant. Content may be a string or an array
of text parts. Image parts are rejected in v0.1. stop accepts a string or an
array; each stop must tokenize to exactly one token in this version.

Non-stream response:

    {
      "id": "chatcmpl-1",
      "object": "chat.completion",
      "created": 1792614698,
      "model": "RustMoE Tiny DS41 Fixture",
      "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "..."},
        "finish_reason": "length"
      }],
      "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
      "rustmoe": {
        "backend": "simulation",
        "simulation": true,
        "ttft_ms": 0.064,
        "prefill_ms": 0.056,
        "decode_ms": 0.014,
        "backend_cache": null,
        "scheduler": {"gpu_hits": 14, "disk_misses": 4, "...": "..."}
      }
    }

Streaming returns text/event-stream. Chunks use chat.completion.chunk. The first
chunk contains the assistant role. Token deltas follow. A final chunk carries
finish_reason. The stream ends with:

    data: [DONE]

Dropping the client connection causes the generation callback to return false;
the blocking generation task stops at the next token boundary.

## POST /v1/completions

Single string prompt only. Supports non-stream and stream. The response object is
text_completion. finish_reason is stop or length.

## POST /v1/cache/reset

Clears scheduler counters, GPU/host cache counters, and prefetch state. It does
not reset lifetime Prometheus counters. Returns the new cache snapshot.

## Errors

OpenAI-style error object:

    {"error":{"message":"...","type":"invalid_request_error","code":"inference_error"}}

Common status codes:

- 400: empty prompt, context overflow, invalid sampler, unsupported content,
  multi-token stop;
- 401: missing/invalid bearer token;
- 404: model id mismatch;
- 500: backend or scheduler failure.

Context overflow is reported, not silently truncated. Prompt and max_tokens
must fit inside the configured context.

## OpenAI and DeepSeek client compatibility

The surface is OpenAI Chat Completions compatible for text. An OpenAI SDK or a
DeepSeek client that speaks the same protocol can use:

    base_url = http://127.0.0.1:8080/v1
    # DeepSeek-style clients may also use http://127.0.0.1:8080
    api_key  = local            # any value unless --api-key is configured
    model    = deepseek-v4.1-flash-1017   # deepseek-chat is also accepted

Supported:

- POST /v1/chat/completions and POST /v1/completions
- DeepSeek-style POST /chat/completions and POST /completions (no /v1)
- stream=false and stream=true (SSE chunks plus [DONE])
- system, user, and assistant text messages
- temperature, top_p, top_k, min_p, repetition_penalty
- max_tokens and max_completion_tokens, seed
- stop strings that tokenize to exactly one token
- model aliases: deepseek-chat, deepseek-reasoner, deepseek-v4.1,
  deepseek-v4.1-flash, deepseek

Not implemented:

- tool/function calling and the tool role
- DeepSeek reasoning_content, thinking mode, and reasoning_effort
- image, audio, or non-text content parts
- response_format / JSON mode enforcement
- logprobs, top_logprobs, n > 1, frequency_penalty, presence_penalty
- /v1/responses, /v1/embeddings, /v1/images, /v1/audio
- the full DeepSeek-V4.1 Jinja template and DSML tool protocol

The aliases route to the same loaded DeepSeek-V4.1 weights. deepseek-reasoner is
accepted for client compatibility but does not return a separate reasoning
channel and is not a DeepSeek-R1 model.

## Chat format

Simulation mode formats:

    role: content
    ...
    assistant:

Real llama.cpp mode uses a basic DeepSeek-V4.1 text chat encoder:

    <BOS><User>message<Assistant></think>

It supports system/user/assistant text turns. It does not implement thinking
mode, reasoning budgets, DSML tool calls, images, or tool results. Use
/v1/completions for exact raw prompts when a chat template is not appropriate.
The vendor-provided Jinja template is not evaluated by RustMoE.
