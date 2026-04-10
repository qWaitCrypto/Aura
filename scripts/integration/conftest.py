from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aura.runtime.engine import Engine, build_engine_for_session  # noqa: E402
from aura.runtime.event_bus import EventBus  # noqa: E402
from aura.runtime.ids import new_id  # noqa: E402
from aura.runtime.llm.config import ModelConfig  # noqa: E402
from aura.runtime.llm.types import (  # noqa: E402
    ContextManagementConfig,
    CredentialRef,
    ModelCapabilities,
    ModelLimits,
    ModelProfile,
    ModelRole,
    ProviderKind,
)
from aura.runtime.project import RuntimePaths  # noqa: E402
from aura.runtime.stores import FileApprovalStore, FileArtifactStore, FileEventLogStore, FileSessionStore  # noqa: E402
from aura.runtime.tools.runtime import ToolApprovalMode  # noqa: E402


@dataclass(frozen=True, slots=True)
class TestRuntime:
    project_root: Path
    paths: RuntimePaths
    session_id: str
    artifact_store: FileArtifactStore
    session_store: FileSessionStore
    approval_store: FileApprovalStore
    event_log_store: FileEventLogStore
    event_bus: EventBus
    engine: Engine


def _default_model_config(*, context_limit_tokens: int = 2048, auto_compact_threshold_ratio: float | None = None) -> ModelConfig:
    profile = ModelProfile(
        profile_id="main",
        provider_kind=ProviderKind.OPENAI_COMPATIBLE,
        base_url="",
        model_name="stub",
        credential_ref=CredentialRef(kind="inline", identifier=""),
        timeout_s=10,
        default_params={},
        capabilities=ModelCapabilities(supports_tools=True, supports_streaming=False),
        tags=set(),
        limits=ModelLimits(context_limit_tokens=int(context_limit_tokens), max_output_tokens=512),
        context_management=ContextManagementConfig(auto_compact_threshold_ratio=auto_compact_threshold_ratio),
    )
    cfg = ModelConfig(
        profiles={"main": profile},
        role_pointers={ModelRole.MAIN: "main", ModelRole.EXTRACT: "main"},
    )
    cfg.validate_consistency()
    return cfg


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def make_runtime(tmp_path: Path) -> Callable[..., TestRuntime]:
    def _make_runtime(
        *,
        tools_enabled: bool,
        approval_mode: ToolApprovalMode = ToolApprovalMode.TRUSTED,
        model_config: ModelConfig | None = None,
        session_id: str | None = None,
    ) -> TestRuntime:
        project_root = tmp_path
        paths = RuntimePaths.for_project(project_root)
        # Ensure `.aura/` exists so code that discovers/creates subpaths behaves consistently.
        paths.system_dir.mkdir(parents=True, exist_ok=True)

        artifact_store = FileArtifactStore(paths.artifacts_dir)
        session_store = FileSessionStore(paths.sessions_dir)
        approval_store = FileApprovalStore(paths.state_dir / "approvals")
        event_log_store = FileEventLogStore(paths.events_dir, artifact_store=artifact_store, session_store=session_store)
        event_bus = EventBus(event_log_store=event_log_store)

        sid = session_id or new_id("sess")
        try:
            session_store.get_session(sid)
        except FileNotFoundError:
            session_store.create_session(
                {
                    "session_id": sid,
                    "project_ref": str(project_root),
                    "mode": "chat",
                    "tool_approval_mode": approval_mode.value,
                }
            )

        cfg = model_config or _default_model_config()
        engine = build_engine_for_session(
            project_root=project_root,
            session_id=sid,
            event_bus=event_bus,
            session_store=session_store,
            event_log_store=event_log_store,
            artifact_store=artifact_store,
            approval_store=approval_store,
            model_config=cfg,
            tools_enabled=tools_enabled,
        )
        if engine.tool_runtime is not None:
            engine.tool_runtime.set_approval_mode(approval_mode)
        engine.load_history_from_events()
        engine.apply_memory_summary_retention()

        return TestRuntime(
            project_root=project_root,
            paths=paths,
            session_id=sid,
            artifact_store=artifact_store,
            session_store=session_store,
            approval_store=approval_store,
            event_log_store=event_log_store,
            event_bus=event_bus,
            engine=engine,
        )

    return _make_runtime


@pytest.fixture
def default_model_config() -> Callable[..., ModelConfig]:
    return _default_model_config
