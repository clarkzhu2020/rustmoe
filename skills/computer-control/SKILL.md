---
name: computer-control
description: Operate a Windows desktop from the shell: launch apps, inspect processes, take screenshots, and use Chrome headless for web UI work.
---

# Computer control

The shell runs with the server user's permissions. Use PowerShell and installed apps.

Examples:

    Get-Process | Sort-Object CPU -Descending | Select-Object -First 10
    Get-ChildItem -Force
    Start-Process notepad
    & "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless --disable-gpu --screenshot=shot.png https://example.com
    & "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless --disable-gpu --dump-dom https://example.com

Rules:

- Prefer non-destructive commands and a dedicated project directory.
- Do not close or modify unrelated user applications.
- Ask before installing system-wide software or changing machine settings.
- A screenshot file is a deliverable; record its path. You cannot visually inspect it unless a vision tool is available.
- For repeatable browser automation, use Playwright if installed, or Chrome headless for DOM/screenshot checks.
