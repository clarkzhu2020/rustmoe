# Skills

RustMoE Dev Agent loads skill packs from the skills directory:

    skills/<name>/SKILL.md

The default is skills/ under the server working directory. Override with:

    --skills-dir F:\rustmoe-skills
    set RUSTMOE_SKILLS_DIR=F:\rustmoe-skills

At the start of each agent run, the server builds a catalog of skill names and
descriptions and adds it to the system prompt. The model loads the full body with
the skill tool:

    {"tool":"skill","args":{"name":"web-research"}}

## Installed skills

Custom workflow packs in this repository:

| Skill | Purpose |
|---|---|
| web-research | Search Bing, fetch pages, save cited notes |
| comfyui-video | Queue and monitor local ComfyUI image/video workflows |
| computer-control | PowerShell, processes, Chrome headless screenshots/DOM |
| website-maintenance | Inspect, edit, test, and deploy an existing site |
| project-delivery | Plan/architecture/code/UI/docs/deploy checklist |

DeepSeek Harness packs imported under skills/dsh/:

- agent-experience
- cordis-composition-reference
- cordis-plugin-development
- editing-cordis-compositions
- diagnose-windows-sandbox-acl
- office
- office-docx
- office-export-pdf
- office-pptx
- office-recalculate
- office-render
- office-xlsx

17 skills were reported by GET /v1/agent/skills in the verification run.

## What a skill can do

A SKILL.md is instructions plus optional helper scripts and references. It does
not grant a new operating-system capability by itself. The agent executes:

- file tools under the agent root
- PowerShell via shell (when --agent-allow-shell)
- helper scripts in the skill directory

So web-research works because it includes a Node search/fetch helper and the
shell is enabled. comfyui-video works only when ComfyUI is running at
http://127.0.0.1:8188 and the workflow JSON is valid.

## Add a skill

    skills/my-workflow/
      SKILL.md
      helper.py
      references/

SKILL.md:

    ---
    name: my-workflow
    description: When to use this skill and what it produces.
    ---

    # Steps

    1. ...
    2. Run: python "$RUSTMOE_SKILLS_DIR/my-workflow/helper.py"

Restart is not required: skills are rescanned for each agent run.

## API

    GET /v1/agent/skills
    GET /v1/agent/skills/{name}

The first returns name, description, path. The second returns the full SKILL.md.

## Importing DeepSeek Harness skills

The skills/dsh tree was extracted from the installed DeepSeek Harness
app.asar. It is a copy of the instruction packs only. Some packs expect tools
that live in DSH itself:

- office-* expects the bundled Python/Office libraries and LibreOffice Kit.
- cordis-* expects the DSH Cordis host/plugin runtime.
- diagnose-windows-sandbox-acl expects its PowerShell diagnostic script.

The agent can still read these instructions and use shell/Python for parts of the
workflow, but they are not automatic DSH plugins. Review each skill before
relying on it.

## Security

- Skills are untrusted local content. Do not install packs from untrusted sources.
- Helper scripts run with the server user's permissions.
- Network tools can reach the internet; ComfyUI and local services are reachable
  from the same machine.
- Keep --agent-root scoped to a dedicated workspace.
