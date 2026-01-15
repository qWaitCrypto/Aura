from __future__ import annotations

import time
import uuid


def new_id(prefix: str) -> str:
    ts = time.time_ns()
    rand = uuid.uuid4().hex
    return f"{prefix}_{ts:016x}_{rand}"


def now_ts_ms() -> int:
    return int(time.time() * 1000)

