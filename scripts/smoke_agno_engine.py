from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aura.runtime.approval import ApprovalStatus
from aura.runtime.engine import build_engine_for_session
from aura.runtime.event_bus import EventBus
from aura.runtime.ids import new_id, now_ts_ms
from aura.runtime.llm.config_io import load_model_config_layers_for_dir
from aura.runtime.llm.openai_stub_server import OpenAIStubServer
from aura.runtime.project import RuntimePaths
from aura.runtime.protocol import ArtifactRef, EventKind, Op, OpKind
from aura.runtime.stores.fs import FileApprovalStore, FileArtifactStore, FileEventLogStore, FileSessionStore
from aura.runtime.tools.runtime import ToolApprovalMode


def _write_models_json(models_path: Path, *, base_url: str) -> None:
    models_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "default_profile": "main",
        "profiles": {
            "main": {
                "provider_kind": "openai_compatible",
                "base_url": base_url,
                "model": "stub",
                "api_key": "",
                "timeout_s": 30,
                "capabilities": {"supports_tools": True, "supports_streaming": True},
            }
        },
    }
    models_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    project_root = PROJECT_ROOT

    # Force the agno engine backend for this smoke run.
    os.environ["AURA_ENGINE"] = "agno"

    paths = RuntimePaths.for_project(project_root)
    for d in [
        paths.config_dir,
        paths.sessions_dir,
        paths.events_dir,
        paths.artifacts_dir,
        paths.state_dir / "approvals",
        paths.index_dir,
        paths.cache_dir,
        paths.tmp_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        with OpenAIStubServer(host="127.0.0.1", port=0) as stub:
            _write_models_json(paths.config_dir / "models.json", base_url=stub.base_url)

            layers = load_model_config_layers_for_dir(project_root, require_project=True)
            model_config = layers.merged()

            artifact_store = FileArtifactStore(paths.artifacts_dir)
            session_store = FileSessionStore(paths.sessions_dir)
            approval_store = FileApprovalStore(paths.state_dir / "approvals")
            event_log_store = FileEventLogStore(paths.events_dir, artifact_store=artifact_store, session_store=session_store)
            event_bus = EventBus(event_log_store=event_log_store)

            session_id = session_store.create_session(
                {
                    "project_ref": str(project_root),
                    "mode": "chat",
                    "tool_approval_mode": ToolApprovalMode.STANDARD.value,
                }
            )

            engine = build_engine_for_session(
                project_root=project_root,
                session_id=session_id,
                event_bus=event_bus,
                session_store=session_store,
                event_log_store=event_log_store,
                artifact_store=artifact_store,
                approval_store=approval_store,
                model_config=model_config,
                tools_enabled=True,
            )
            if engine.tool_runtime is not None:
                engine.tool_runtime.set_approval_mode(ToolApprovalMode.STANDARD)
            engine.load_history_from_events()
            engine.apply_memory_summary_retention()

            turn_id = new_id("turn")
            request_id = new_id("req")
            engine.handle(
                Op(
                    kind=OpKind.CHAT.value,
                    payload={"text": "smoke test: run a tool"},
                    session_id=session_id,
                    request_id=request_id,
                    timestamp=now_ts_ms(),
                    turn_id=turn_id,
                ),
                timeout_s=30,
            )

            # Auto-approve any approval requests created by the tool loop.
            while True:
                pending = approval_store.list(session_id=session_id, status=ApprovalStatus.PENDING)
                if not pending:
                    break
                rec = pending[0]
                engine.handle(
                    Op(
                        kind=OpKind.APPROVAL_DECISION.value,
                        payload={"approval_id": rec.approval_id, "decision": "approve"},
                        session_id=session_id,
                        request_id=new_id("req"),
                        timestamp=now_ts_ms(),
                        turn_id=rec.turn_id,
                    ),
                    timeout_s=30,
                )

            event_bus.flush(session_id=session_id)

            last_output_ref: dict[str, Any] | None = None
            for ev in event_log_store.read(session_id):
                if ev.kind == EventKind.LLM_RESPONSE_COMPLETED.value:
                    raw = ev.payload.get("output_ref")
                    if isinstance(raw, dict):
                        last_output_ref = raw

            if last_output_ref is not None:
                text = artifact_store.get(ArtifactRef.from_dict(last_output_ref)).decode("utf-8", errors="replace")
                print("\n=== assistant output ===\n")
                print(text)

            print("\n=== smoke run ===")
            print(f"session_id: {session_id}")
            print(f"events: {paths.events_dir / f'{session_id}.jsonl'}")
            print(f"artifacts: {paths.artifacts_dir}")
            return 0
    except PermissionError as e:
        print(f"smoke_agno_engine: could not start OpenAIStubServer (permission denied): {e}")
        print("Tip: run this smoke test outside a restricted sandbox.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
