---
name: website-maintenance
description: Inspect, edit, test, and deploy an existing website with safe backups and verification.
---

# Website maintenance

1. Read the project README and identify the framework, package manager, build, test, and deploy commands.
2. Make a backup or work on a branch before editing.
3. Reproduce the current issue; capture the failing command or URL.
4. Edit the smallest relevant files.
5. Run build, tests, and a local smoke check.
6. If a deploy step is requested, verify credentials and a rollback path first.
7. Write a change summary with files, commands, URLs, and remaining risks.

Useful commands:

    git status --short
    git diff
    npm test
    npm run build
    curl -I http://127.0.0.1:3000
    docker compose config
