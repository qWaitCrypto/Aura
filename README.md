# Aura

**A developer-experience-first AI agent framework for building controllable, auditable agents.**

Aura originated from [Novelaire](https://github.com/qWaitCrypto/novelaire), a creative writing assistant, and evolved into a general-purpose agent platform. It is now powered by [Agno](https://github.com/agno-ai/agno), combining Agno's multi-model orchestration with Aura's focus on transparency, version control, and local-first workflows.

---

## Why Aura?

While many agent frameworks focus on ease-of-use or cloud integration, **Aura prioritizes developer control**:

- **Audit Trails**: Every event (LLM calls, tool executions, approvals) logged to `.aura/events/` as append-only JSONL
- **Project-as-Code**: Agent state lives in `.aura/` — committed to git, sharable, reproducible
- **Approval Policies**: Fine-grained control over tool execution with inspection, approval, and denial workflows
- **Multi-Surface**: Same engine powers CLI, Web, Plugin, or Cloud deployments via a simple `Surface` protocol
- **Cross-Process Recovery**: Pause/resume workflows across machine restarts using run snapshots

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/qWaitCrypto/aura.git
cd aura

# Initialize Aura in your workspace
aura init .

# Start a chat session
aura chat
```

> **Note**: `.aura/` is created in your **workspace**, not in the Aura repository. This keeps your project state separate from the framework code.

---

## Core Concepts

### 1. **Session & Events**
Every conversation is a session stored in `.aura/sessions/<session_id>.json`. All events (requests, responses, tool calls) append to `.aura/events/<session_id>.jsonl`.

### 2. **Tools & Approval**
Aura ships with built-in tools (file operations, shell, web, specs, skills). Each tool call can be:
- **Auto-approved** (low-risk, like reading files)
- **Require approval** (high-risk, like shell commands)
- **Denied** (policy violations)

### 3. **Skills**
Reusable capabilities defined as markdown files in `.aura/skills/`. Skills combine instructions with optional tool allowlists.

```markdown
---
name: code-review
description: Automated code review with security checks
allowed_tools:
  - project__read_text
  - project__search_text
---
# Instructions
...
```

### 4. **Subagents**
Delegate subtasks to isolated agents with scoped tool access and guardrails (max turns, max tool calls).

### 5. **Compaction**
Automatically summarize long conversations when context approaches limits, preserving memory while trimming history.

### 6. **MCP Integration**
Support for [Model Context Protocol](https://modelcontextprotocol.io/) servers via `.aura/config/mcp.json`.

### 7. **RAG (Optional)**
Project knowledge retrieval using BM25, embeddings, or hybrid search. Disabled by default; opt-in via configuration.

---

## Multi-Surface Architecture

Aura's engine is surface-agnostic. The same runtime can power:

- **CLI**: `aura chat` (current implementation)
- **Web**: FastAPI + WebSocket (stub provided)
- **Cloud**: Stateless REST API with cross-process recovery (stub provided)
- **Plugin**: VS Code / Cursor extensions (future)

See `aura/runtime/surface.py` for the interface.

---

## Directory Structure

### Your Workspace (created by `aura init`)

```
<your-workspace>/
└── .aura/
    ├── config/
    │   └── mcp.json        # MCP server configs
    ├── sessions/           # Chat sessions
    ├── events/             # Event logs (JSONL)
    ├── runs/               # Paused run snapshots
    ├── skills/             # Loaded skills
    ├── knowledge/          # RAG documents (optional)
    └── vectordb/           # Vector DB storage (optional)
```

### Aura Repository

```
aura/                       # Framework code
├── runtime/
│   ├── engine_agno_async.py  # Async-first engine
│   ├── tools/                # Built-in tools
│   ├── subagents/            # Subagent runner
│   ├── skills.py             # Skill store
│   ├── knowledge/            # RAG module
│   └── mcp/                  # MCP integration
├── surfaces/               # Surface implementations
└── cli.py                  # CLI entry point
```

---

## How Aura Extends Agno

| Feature | Agno Provides | Aura Adds |
|---------|---------------|-----------|
| **Multi-model orchestration** | ✅ Agent, Tools, Knowledge | File-based state (`.aura/`), audit trails |
| **Tool execution** | ✅ Function calling | Approval policies, inspection, deny rules |
| **Memory** | ✅ `AgentMemory` | Explicit compaction, session snapshots |
| **MCP** | ✅ `MCPTools` | Config-driven server management |
| **Skills** | ✅ `LocalSkills` | Project-local skill store, nested categories |
| **RAG** | ✅ `Knowledge` | Opt-in, offline BM25, pluggable backends |
| **Subagents** | ✅ Multi-agent (Team) | Single-agent isolation with approval bypass prevention |

---

## Development Status

**Current**: v0.2 (Agno-powered, fully async)

Recent milestones:
- ✅ Migrated to Agno runtime (from custom engine)
- ✅ Async-first engine with pause/resume
- ✅ Cross-process run recovery
- ✅ MCP integration
- ✅ Optional RAG module
- ✅ Multi-surface abstraction
- ✅ Integration tests (9/10 passing)

---

## Philosophy

**Aura is designed for developers building AI agents, not for end-users chatting with AI.**

We believe:
- **Transparency over magic**: Every decision is logged and reviewable
- **Local-first**: Your data lives in git, not a cloud database
- **Deliberate control**: Agents should require approval for risky actions
- **Composability**: Skills, tools, and surfaces are building blocks, not black boxes

---

## Origins

Aura started as the runtime for **Novelaire**, a creative writing assistant. As we extracted the agent logic from domain-specific features, we recognized the need for a general-purpose framework that preserved the audit trails, approval workflows, and project-centric design that made Novelaire reliable.

After evaluating existing frameworks, we chose to build on **Agno** for its strong multi-model support and extensibility, while preserving Aura's unique DX philosophy.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

---

## License

[MIT](LICENSE)

---

## Keywords for Discoverability

To help people find Aura:

**GitHub Topics** (add to repo settings):
- `ai-agent`
- `llm-framework`
- `agno`
- `developer-tools`
- `local-first`
- `audit-trail`
- `mcp`
- `rag`
- `multi-agent`
- `python-agent`

**README Keywords** (already embedded above):
- AI agent framework
- LLM orchestration
- Developer-first agent
- Audit trail
- Project-as-code
- Approval workflow
- Multi-surface agent
- Local-first AI
- Cross-process recovery
- Model Context Protocol

**Social/SEO**:
- Blog post: "Building Transparent AI Agents with Aura"
- Reddit: r/LocalLLaMA, r/MachineLearning
- Hacker News: "Show HN: Aura – Developer-first AI agent framework"
- Twitter/X: Tag #AI, #LLM, #AgentFramework, #LocalFirst