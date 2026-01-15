from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .ids import now_ts_ms
from .stores import SessionStore


class StepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class PlanItem:
    step: str
    status: StepStatus

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "status": self.status.value}

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "PlanItem":
        step = raw.get("step")
        status = raw.get("status")
        if not isinstance(step, str) or not step.strip():
            raise ValueError("PlanItem.step must be a non-empty string.")
        if not isinstance(status, str) or not status:
            raise ValueError("PlanItem.status must be a non-empty string.")
        try:
            st = StepStatus(status)
        except ValueError as e:
            raise ValueError(f"Invalid plan status: {status!r}") from e
        return PlanItem(step=step.strip(), status=st)


@dataclass(frozen=True, slots=True)
class PlanState:
    plan: list[PlanItem]
    explanation: str | None = None
    updated_at: int | None = None


def validate_plan(items: list[PlanItem]) -> None:
    in_progress = sum(1 for t in items if t.status is StepStatus.IN_PROGRESS)
    if in_progress > 1:
        raise ValueError("Plan can contain at most one item with status='in_progress'.")


class PlanStore:
    """
    Session-scoped plan persistence (Codex-style update_plan + Goose-style persistence).

    Reference storage: SessionStore JSON meta under keys:
    - plan: list[{step,status}]
    - plan_explanation: str?
    - plan_updated_at: int (ms)
    """

    def __init__(self, *, session_store: SessionStore, session_id: str) -> None:
        self._session_store = session_store
        self._session_id = session_id

    def get(self) -> PlanState:
        meta = self._session_store.get_session(self._session_id)
        raw_plan = meta.get("plan")
        raw_expl = meta.get("plan_explanation")
        raw_updated = meta.get("plan_updated_at")

        items: list[PlanItem] = []
        if isinstance(raw_plan, list):
            for item in raw_plan:
                if not isinstance(item, dict):
                    continue
                try:
                    items.append(PlanItem.from_dict(item))
                except ValueError:
                    continue

        explanation = raw_expl if isinstance(raw_expl, str) and raw_expl.strip() else None
        updated_at = raw_updated if isinstance(raw_updated, int) else None
        return PlanState(plan=items, explanation=explanation, updated_at=updated_at)

    def set(self, items: list[PlanItem], *, explanation: str | None = None) -> None:
        validate_plan(items)
        payload: dict[str, Any] = {
            "plan": [t.to_dict() for t in items],
            "plan_updated_at": now_ts_ms(),
            # Always write explanation key so re-plans can clear it by passing null.
            "plan_explanation": explanation,
        }
        self._session_store.update_session(self._session_id, payload)

