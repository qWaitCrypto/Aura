from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..plan import PlanItem, PlanStore, StepStatus, validate_plan


@dataclass(frozen=True, slots=True)
class UpdatePlanTool:
    store: PlanStore
    name: str = "update_plan"
    description: str = (
        "Updates the task plan.\n"
        "Provide an optional explanation and a list of plan items, each with a step and status.\n"
        "At most one step can be in_progress at a time."
    )
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "explanation": {"type": "string"},
                "plan": {
                    "type": "array",
                    "description": "The list of steps",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string"},
                            "status": {
                                "type": "string",
                                "description": "One of: pending, in_progress, completed",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["step", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["plan"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root) -> dict[str, Any]:
        del project_root
        raw_plan = args.get("plan")
        explanation = args.get("explanation")

        if not isinstance(raw_plan, list):
            raise ValueError("Missing or invalid 'plan' (expected list).")
        if explanation is not None and not isinstance(explanation, str):
            raise ValueError("Invalid 'explanation' (expected string).")

        items: list[PlanItem] = []
        for raw in raw_plan:
            if not isinstance(raw, dict):
                raise ValueError("Invalid plan item (expected object).")
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
            items.append(PlanItem(step=step.strip(), status=st))

        validate_plan(items)
        self.store.set(items, explanation=explanation)
        return {"ok": True, "message": "Plan updated", "steps": len(items)}

