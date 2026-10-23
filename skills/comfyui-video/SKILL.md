---
name: comfyui-video
description: Drive a local ComfyUI server to queue, monitor, and download image or video workflows.
---

# ComfyUI workflows

Use the shell tool and the helper in this skill directory.

Check the server:

    node "$RUSTMOE_SKILLS_DIR/comfyui-video/comfyui.mjs" status

Queue a workflow JSON:

    node "$RUSTMOE_SKILLS_DIR/comfyui-video/comfyui.mjs" queue workflow.json

Inspect a prompt:

    node "$RUSTMOE_SKILLS_DIR/comfyui-video/comfyui.mjs" history <prompt_id>

Rules:

- The default endpoint is http://127.0.0.1:8188; override with COMFYUI_URL.
- Save downloaded videos/images into the project under assets/.
- For video workflows, use the workflow's SaveVideo/VHS_VideoCombine node output; poll history until the output entries appear.
- Do not queue unknown workflows blindly; validate node types and model filenames first.
- Record the workflow JSON and seed in the project so the result is reproducible.
