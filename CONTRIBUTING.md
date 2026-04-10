# Contributing to Aura

Thanks for improving Aura. This project is still evolving quickly, so small, focused changes are easier to review and keep stable.

## Development setup

```bash
git clone https://github.com/qWaitCrypto/Aura.git
cd Aura

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the test suite

```bash
python -m pytest -q -p no:cacheprovider scripts/integration
```

Useful local checks:

```bash
python -m build
python -m aura --version
```

## Project layout

- `aura/`: CLI, runtime engine, tool system, provider adapters, and storage layers.
- `scripts/integration/`: integration tests for approvals, sessions, recovery, MCP, and compaction.
- `scripts/`: smoke utilities for manual end-to-end checks.

## Coding guidelines

- Keep features local and composable. Aura prefers narrow modules over framework-wide indirection.
- Preserve the local-first model: project state should live under `.aura/`, not hidden services.
- Treat risky actions as approval-gated by default.
- Avoid adding dependencies unless they materially improve the developer workflow.

## Commit style

Use short conventional commits:

- `feat(runtime): add dag scheduler`
- `fix(cli): avoid duplicate approval prompts`
- `docs: tighten quick start`

Keep messages natural. Do not turn the subject line into a mini changelog.

## Pull requests

Before opening a PR:

1. Run the integration suite.
2. Confirm packaging still works with `python -m build`.
3. Update `README.md` or `CONTRIBUTING.md` when the developer workflow changes.
4. Keep the diff focused; separate refactors from behavior changes when possible.
