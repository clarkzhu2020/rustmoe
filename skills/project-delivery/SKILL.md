---
name: project-delivery
description: Deliver a complete production-oriented project with plan, architecture, code, UI, docs, deployment, and verification.
---

# Project delivery

Required artifacts unless the task says otherwise:

- PLAN.md: scope, milestones, acceptance criteria.
- ARCHITECTURE.md: components, data model, APIs, security, failure modes.
- Source code with tests.
- README.md: setup, run, test, troubleshooting.
- DEPLOYMENT.md and Dockerfile/docker-compose or platform-specific deployment.
- .env.example with no secrets.
- UI implementation and responsive behavior.

Before finish:

- Run the build and tests.
- Check the app starts and the main path works.
- Verify no secrets or absolute local paths are committed.
- List known limitations.
