from __future__ import annotations

import queue
import shutil
import sys
import threading
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


class UIEventKind(str, Enum):
    SESSION_STARTED = "session_started"
    PROMPT_READY = "prompt_ready"
    USER_SUBMITTED = "user_submitted"

    CLEAR_SCREEN = "clear_screen"

    LLM_REQUEST_STARTED = "llm_request_started"
    THINKING_DELTA = "thinking_delta"
    THINKING_END = "thinking_end"
    ASSISTANT_DELTA = "assistant_delta"
    ASSISTANT_COMPLETED = "assistant_completed"

    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"

    PLAN_UPDATED = "plan_updated"

    PROGRESS = "progress"
    WARNING = "warning"
    LOG = "log"

    ERROR_RAISED = "error_raised"
    CANCELLED = "cancelled"

    EXIT_REQUESTED = "exit_requested"


@dataclass(frozen=True, slots=True)
class UIEvent:
    kind: UIEventKind
    payload: dict


class ConsoleUI:
    """
    Single-writer, event-driven console UI (line-mode).

    - Only the renderer thread writes to stdout.
    - All other threads call `emit()` to enqueue UIEvents.
    - A built-in tick loop drives the spinner without a separate writer thread.
    """

    def __init__(self, *, stream=None, enable_color: bool = True) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._ansi = bool(getattr(self._stream, "isatty", lambda: False)())
        self._enable_color = enable_color and self._ansi

        self._q: "queue.Queue[UIEvent]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Render state
        self._waiting_for_llm = False
        self._assistant_open = False
        self._assistant_last_newline = True
        self._assistant_nl_run = 0

        self._spinner_frame = 0
        self._thinking_buf = ""
        self._thinking_max_chars = 120
        self._last_spinner_paint = 0.0
        self._plain_waiting_printed = False
        self._saw_thinking = False
        self._printed_think_preview = False
        self._spinner_label = "Thinking"

        # Tool log buffering (Codex/Goose-style): group tool calls into a compact block
        # and flush before the next LLM request starts.
        self._pending_tool_items: list[tuple[str, str]] = []
        self._pending_tool_omitted = 0

        # Interop with full-screen terminal UIs (e.g. prompt_toolkit dialogs).
        self._io_lock = threading.RLock()
        self._suspend_lock = threading.Lock()
        self._suspend_count = 0
        self._deferred_events: list[UIEvent] = []

    @contextmanager
    def suspend(self) -> Iterable[None]:
        """
        Temporarily stop rendering to stdout while a full-screen terminal UI runs.

        The renderer thread continues consuming events but buffers them until resumed.
        """

        first = False
        with self._suspend_lock:
            self._suspend_count += 1
            first = self._suspend_count == 1

        if first:
            with self._io_lock:
                self._stop_waiting(clear_line=True)
                self._ensure_newline_if_streaming()

        try:
            yield
        finally:
            deferred: list[UIEvent] = []
            last = False
            with self._suspend_lock:
                if self._suspend_count > 0:
                    self._suspend_count -= 1
                last = self._suspend_count == 0
                if last and self._deferred_events:
                    deferred = self._deferred_events
                    self._deferred_events = []

            if last and deferred:
                for ev in deferred:
                    self.emit(ev)

    def _tool_category(self, tool_name: str) -> str:
        if tool_name in {
            "project__read_text",
            "project__read_text_many",
            "project__search_text",
            "project__list_dir",
            "project__glob",
            "project__text_stats",
            "session__search",
            "web__fetch",
            "web__search",
            "mcp__list_servers",
            "mcp__list_tools",
        } or tool_name.startswith("skill__"):
            return "Explored"
        if tool_name in {"project__apply_patch", "project__apply_edits"}:
            return "Edited"
        if tool_name == "update_plan":
            return "Planned"
        if tool_name.startswith("spec__"):
            return "Spec"
        if tool_name == "session__export":
            return "Ran"
        if tool_name == "shell__run":
            return "Ran"
        return "Tools"

    def _queue_tool_item(self, *, category: str, line: str) -> None:
        self._pending_tool_items.append((category, line))
        max_items = 40
        if len(self._pending_tool_items) > max_items:
            self._pending_tool_items = self._pending_tool_items[-max_items:]
            self._pending_tool_omitted += 1

    def _flush_tool_items(self) -> None:
        if not self._pending_tool_items:
            return
        self._ensure_newline_if_streaming()

        order = ["Explored", "Edited", "Planned", "Spec", "Ran", "Tools"]
        groups: dict[str, list[str]] = {}
        for cat, line in self._pending_tool_items:
            groups.setdefault(cat, []).append(line)

        for cat in order:
            lines = groups.get(cat)
            if not lines:
                continue
            self._println_dim(f"• {cat}")
            for i, line in enumerate(lines):
                prefix = "  └ " if i == 0 else "    "
                self._println_dim(prefix + line)

        if self._pending_tool_omitted:
            self._println_dim(f"  ... ({self._pending_tool_omitted} earlier tool events omitted)")

        self._pending_tool_items.clear()
        self._pending_tool_omitted = 0

    def _println_multiline(self, msg: str, *, dim: bool, prefix: str | None = None) -> None:
        if msg == "":
            self._println()
            return
        lines = msg.splitlines()
        if msg.endswith("\n"):
            lines.append("")
        for line in lines:
            out = line if prefix is None else prefix + line
            if dim:
                self._println_dim(out)
            else:
                self._println(out)

    # --- lifecycle ---
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._render_loop, name="aura-ui", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        self._stop.set()
        self.emit(UIEvent(UIEventKind.EXIT_REQUESTED, {"code": 0}))
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=join_timeout_s)

    def emit(self, event: UIEvent) -> None:
        # Never block producers; if the queue explodes something else is wrong.
        try:
            self._q.put_nowait(event)
        except Exception:
            pass

    # --- high level helpers (optional) ---
    def print_header(self, *, session_id: str) -> None:
        self.emit(UIEvent(UIEventKind.SESSION_STARTED, {"session_id": session_id}))

    def print_progress(self, message: str) -> None:
        self.emit(UIEvent(UIEventKind.PROGRESS, {"label": message}))

    # --- rendering ---
    def _render_loop(self) -> None:
        tick_interval_s = 0.08
        while not self._stop.is_set():
            try:
                ev = self._q.get(timeout=tick_interval_s)
                with self._suspend_lock:
                    suspended = self._suspend_count > 0
                if suspended and ev.kind is not UIEventKind.EXIT_REQUESTED:
                    with self._suspend_lock:
                        self._deferred_events.append(ev)
                    continue
                self._handle_event(ev)
            except queue.Empty:
                with self._suspend_lock:
                    suspended = self._suspend_count > 0
                if not suspended:
                    self._tick()
            except Exception:
                # UI must not crash the process.
                pass

        # Final cleanup: clear spinner line if needed.
        try:
            self._clear_spinner_line()
        except Exception:
            pass

    def _tick(self) -> None:
        if not self._waiting_for_llm:
            return
        if not self._ansi:
            return
        now = time.monotonic()
        if (now - self._last_spinner_paint) < 0.06:
            return
        self._last_spinner_paint = now
        self._paint_spinner()

    def _handle_event(self, ev: UIEvent) -> None:
        k = ev.kind
        p = ev.payload

        if k is UIEventKind.SESSION_STARTED:
            self._println_dim(f"Session: {p.get('session_id','')}".strip())
            self._println_dim("Commands: /help /clear /perm /model /exit. Ctrl+C cancels.")
            return

        if k is UIEventKind.PROMPT_READY:
            # UI doesn't draw the prompt itself; the input loop does.
            return

        if k is UIEventKind.USER_SUBMITTED:
            self._ensure_newline_if_streaming()
            self._println_user(str(p.get("text", "")))
            return

        if k is UIEventKind.CLEAR_SCREEN:
            self._stop_waiting(clear_line=True)
            if self._ansi:
                # Clear screen + move cursor home.
                self._write("\x1b[2J\x1b[H")
            else:
                self._println()
            return

        if k is UIEventKind.LLM_REQUEST_STARTED:
            # Flush previous tool block (if any) before starting the next LLM request.
            self._flush_tool_items()
            self._waiting_for_llm = True
            self._thinking_buf = ""
            self._spinner_frame = 0
            self._last_spinner_paint = 0.0
            self._plain_waiting_printed = False
            self._saw_thinking = False
            self._printed_think_preview = False
            self._spinner_label = str(p.get("label") or "Thinking")
            self._paint_spinner()
            return

        if k is UIEventKind.THINKING_DELTA:
            if not self._waiting_for_llm:
                self._waiting_for_llm = True
            s = str(p.get("text", "") or "")
            if s:
                self._thinking_buf = (self._thinking_buf + s)[-self._thinking_max_chars :]
                self._saw_thinking = True
            self._paint_spinner()
            return

        if k is UIEventKind.THINKING_END:
            # Don't force-stop here; the first assistant token will stop it.
            self._paint_spinner()
            return

        if k is UIEventKind.ASSISTANT_DELTA:
            delta = str(p.get("text", "") or "")
            if not delta:
                return
            # Preserve a one-line "thinking" preview if we received it but didn't have time to show it
            # (e.g. non-streaming responses where thinking+answer arrive in the same delta, or very fast first token).
            if self._waiting_for_llm and self._saw_thinking and not self._printed_think_preview:
                snippet = self._thinking_buf.strip().replace("\n", " ")
                if snippet:
                    # Stop the spinner in-place first, then print a stable preview line.
                    self._stop_waiting(clear_line=True)
                    cols = shutil.get_terminal_size((80, 20)).columns
                    preview_width = max(20, min(int(cols * 0.6), 72))
                    preview = self._elide_tail(snippet, preview_width)
                    self._println_dim(f"(think: {preview})")
                    self._printed_think_preview = True
                else:
                    self._stop_waiting(clear_line=True)
            else:
                self._stop_waiting(clear_line=True)
            self._start_assistant_if_needed()
            # Avoid an empty "Assistant:" line when the first chunk begins with newlines.
            if self._assistant_open and delta.startswith("\n"):
                delta = delta.lstrip("\n")
            delta = self._compact_blank_lines(delta)
            if not delta:
                return
            self._write(delta)
            self._assistant_last_newline = delta.endswith("\n")
            return

        if k is UIEventKind.ASSISTANT_COMPLETED:
            self._stop_waiting(clear_line=True)
            self._ensure_newline_if_streaming()
            self._assistant_open = False
            self._assistant_last_newline = True
            return

        if k is UIEventKind.TOOL_CALL_STARTED:
            # Tool calls are summarized on completion and flushed as a grouped block.
            return

        if k is UIEventKind.TOOL_CALL_COMPLETED:
            tool = str(p.get("tool", "tool"))
            summary = str(p.get("summary") or tool)
            status = str(p.get("status") or ("succeeded" if bool(p.get("ok", True)) else "failed"))
            duration_ms = p.get("duration_ms")
            error_code = p.get("error_code")
            error = p.get("error")
            details = p.get("details")

            suffix_parts: list[str] = []
            if isinstance(duration_ms, int) and duration_ms >= 500:
                suffix_parts.append(f"{duration_ms}ms")
            # Special-case approval denial to avoid noisy duplicates like "(cancelled; cancelled; Approval denied.)".
            if str(error_code) == "cancelled" and str(error).strip() == "Approval denied.":
                suffix_parts.append("approval denied")
            else:
                if status and status not in {"succeeded", "ok"}:
                    suffix_parts.append(status)
                if error_code and str(error_code) not in {"", "none"}:
                    # Avoid repeating "cancelled" twice (status + code).
                    if not (status == "cancelled" and str(error_code) == "cancelled"):
                        suffix_parts.append(str(error_code))
                if error:
                    one_line = " ".join(str(error).splitlines()).strip()
                    if one_line:
                        suffix_parts.append(one_line)

            suffix = ""
            if suffix_parts:
                suffix = " (" + "; ".join(suffix_parts) + ")"

            category = self._tool_category(tool)
            suppress_summary = tool == "project__apply_patch" and isinstance(details, list) and details
            if not suppress_summary:
                self._queue_tool_item(category=category, line=summary + suffix)
            if isinstance(details, list):
                for item in details:
                    if not isinstance(item, str):
                        continue
                    s = item.rstrip("\r\n")
                    if s.strip():
                        self._queue_tool_item(category=category, line=s)
            return

        if k is UIEventKind.PLAN_UPDATED:
            self._stop_waiting(clear_line=True)
            self._ensure_newline_if_streaming()

            explanation = str(p.get("explanation") or "").strip()
            raw_plan = p.get("plan")

            self._println_dim("[plan] Updated plan")
            if explanation:
                one_line = " ".join(explanation.splitlines()).strip()
                if one_line:
                    self._println_dim(f"  (why: {self._elide_tail(one_line, 80)})")

            if not isinstance(raw_plan, list) or not raw_plan:
                self._println_dim("  (empty)")
                return

            max_lines = 12
            shown = 0
            for item in raw_plan:
                if shown >= max_lines:
                    break
                if not isinstance(item, dict):
                    continue
                step = str(item.get("step") or "").strip()
                status = str(item.get("status") or "").strip()
                if not step or not status:
                    continue

                if status == "completed":
                    prefix = "[x]"
                    self._println_dim(f"  {prefix} {step}")
                elif status == "in_progress":
                    prefix = "[~]"
                    self._println(self._color(f"  {prefix} {step}", "1;36"))
                else:
                    prefix = "[ ]"
                    self._println_dim(f"  {prefix} {step}")
                shown += 1

            remaining = len([x for x in raw_plan if isinstance(x, dict)]) - shown
            if remaining > 0:
                self._println_dim(f"  ... ({remaining} more)")
            return

        if k is UIEventKind.PROGRESS:
            self._stop_waiting(clear_line=True)
            self._ensure_newline_if_streaming()
            label = str(p.get("label", "") or "")
            detail = str(p.get("detail", "") or "")
            msg = label if not detail else f"{label} {detail}"
            if msg.strip():
                self._println_dim(f"[progress] {msg.strip()}")
            return

        if k is UIEventKind.WARNING:
            self._stop_waiting(clear_line=True)
            self._ensure_newline_if_streaming()
            self._println_yellow(f"[warn] {p.get('message','')}")
            return

        if k is UIEventKind.LOG:
            self._stop_waiting(clear_line=True)
            self._ensure_newline_if_streaming()
            level = str(p.get("level", "info"))
            msg = str(p.get("message", "") or "")
            if msg:
                if level == "approval":
                    self._println_multiline(msg, dim=False)
                else:
                    self._println_multiline(f"[{level}] {msg}", dim=True)
            return

        if k is UIEventKind.ERROR_RAISED:
            # Ensure any completed tool calls are visible before showing the error.
            self._flush_tool_items()
            self._stop_waiting(clear_line=True)
            self._ensure_newline_if_streaming()
            msg = str(p.get("message", "") or "")
            code = str(p.get("code", "") or "")
            prefix = f"[error] {code}: " if code else "[error] "
            self._println_red(prefix + msg)
            self._assistant_open = False
            self._assistant_last_newline = True
            return

        if k is UIEventKind.CANCELLED:
            self._stop_waiting(clear_line=True)
            self._ensure_newline_if_streaming()
            msg = str(p.get("message", "") or "cancelled")
            self._println_yellow(f"[cancel] {msg}")
            self._assistant_open = False
            self._assistant_last_newline = True
            return

        if k is UIEventKind.EXIT_REQUESTED:
            self._stop_waiting(clear_line=True)
            return

    # --- low-level printing ---
    def _color(self, s: str, code: str) -> str:
        if not self._enable_color:
            return s
        return f"\x1b[{code}m{s}\x1b[0m"

    def _write(self, s: str) -> None:
        with self._io_lock:
            self._stream.write(s)
            try:
                self._stream.flush()
            except Exception:
                pass

    def _println(self, s: str = "") -> None:
        self._write(s + "\n")

    def _println_dim(self, s: str) -> None:
        self._println(self._color(s, "2"))

    def _println_red(self, s: str) -> None:
        self._println(self._color(s, "31"))

    def _println_yellow(self, s: str) -> None:
        self._println(self._color(s, "33"))

    def _println_user(self, text: str) -> None:
        prefix = self._color("You: ", "1;32") if self._enable_color else "You: "
        self._println(prefix + text)

    def _start_assistant_if_needed(self) -> None:
        if self._assistant_open:
            return
        prefix = self._color("Assistant: ", "1;36") if self._enable_color else "Assistant: "
        self._write(prefix)
        self._assistant_open = True
        self._assistant_last_newline = False
        self._assistant_nl_run = 0

    def _ensure_newline_if_streaming(self) -> None:
        if not self._assistant_open:
            return
        if not self._assistant_last_newline:
            self._println()
        self._assistant_open = False
        self._assistant_last_newline = True
        self._assistant_nl_run = 0

    def _compact_blank_lines(self, delta: str) -> str:
        # Keep the output compact in line-mode: collapse 2+ newlines into 1 newline.
        out: list[str] = []
        nl_run = self._assistant_nl_run
        for ch in delta:
            if ch == "\r":
                continue
            if ch == "\n":
                if nl_run >= 1:
                    continue
                nl_run += 1
                out.append(ch)
                continue
            nl_run = 0
            out.append(ch)
        self._assistant_nl_run = nl_run
        return "".join(out)

    def _stop_waiting(self, *, clear_line: bool) -> None:
        if not self._waiting_for_llm:
            return
        self._waiting_for_llm = False
        if clear_line:
            self._clear_spinner_line()

    def _clear_spinner_line(self) -> None:
        if not self._ansi:
            return
        self._write("\r\x1b[2K\r")

    def _paint_spinner(self) -> None:
        if not self._waiting_for_llm:
            return
        if not self._ansi:
            if not self._plain_waiting_printed:
                self._println_dim(f"{self._spinner_label}…")
                self._plain_waiting_printed = True
            return
        # "circle" frames; fallback to ASCII if the terminal can't render them is fine.
        frames: Sequence[str] = ("◌", "◍", "●", "◍")
        ch = frames[self._spinner_frame % len(frames)]
        self._spinner_frame += 1
        msg = self._spinner_label
        snippet_raw = self._thinking_buf.strip().replace("\n", " ")

        cols = shutil.get_terminal_size((80, 20)).columns
        max_cols = max(20, int(cols) - 1)

        prefix = f"{ch} {msg}"
        if snippet_raw:
            # Keep the most recent part visible.
            avail = max(0, max_cols - self._display_width(prefix) - 3)
            # Make the thinking snippet intentionally small and stable.
            snippet_target = min(avail, 50)
            snippet = self._elide_tail(snippet_raw, snippet_target)
            line = f"{prefix} ({snippet})" if snippet else f"{prefix}…"
        else:
            line = f"{prefix}…"

        # Ensure the spinner never wraps; wrapping breaks in-place updates.
        line = self._truncate_to_width(line, max_cols)
        # Paint in-place.
        self._write("\r\x1b[2K\r" + line)

    def _display_width(self, s: str) -> int:
        w = 0
        for ch in s:
            if unicodedata.combining(ch):
                continue
            eaw = unicodedata.east_asian_width(ch)
            w += 2 if eaw in {"W", "F"} else 1
        return w

    def _truncate_to_width(self, s: str, width: int) -> str:
        if width <= 0:
            return ""
        if self._display_width(s) <= width:
            return s
        out: list[str] = []
        used = 0
        for ch in s:
            if unicodedata.combining(ch):
                out.append(ch)
                continue
            eaw = unicodedata.east_asian_width(ch)
            cw = 2 if eaw in {"W", "F"} else 1
            if used + cw > width:
                break
            out.append(ch)
            used += cw
        return "".join(out)

    def _elide_tail(self, s: str, width: int) -> str:
        if width <= 0:
            return ""
        if self._display_width(s) <= width:
            return s
        # Keep tail with a leading ellipsis.
        if width == 1:
            return "…"
        target = width - 1
        out_rev: list[str] = []
        used = 0
        for ch in reversed(s):
            if unicodedata.combining(ch):
                out_rev.append(ch)
                continue
            eaw = unicodedata.east_asian_width(ch)
            cw = 2 if eaw in {"W", "F"} else 1
            if used + cw > target:
                break
            out_rev.append(ch)
            used += cw
        return "…" + "".join(reversed(out_rev))


class ThinkTagParser:
    """
    Best-effort streaming parser for common "thinking" tags embedded in TEXT_DELTA streams.

    Many open-source reasoning models expose scratchpad as e.g.:
      - <think> ... </think>
      - <thinking> ... </thinking>

    Produces a sequence of (is_thinking, text) segments.
    """

    def __init__(self, *, pairs: Iterable[tuple[str, str]] | None = None, case_insensitive: bool = True) -> None:
        self._pairs: list[tuple[str, str]] = list(
            pairs
            if pairs is not None
            else [
                ("<think>", "</think>"),
                ("<thinking>", "</thinking>"),
            ]
        )
        self._case_insensitive = case_insensitive
        self._in_think = False
        self._carry = ""
        self._active_end: str | None = None

    @property
    def in_think(self) -> bool:
        return self._in_think

    def feed(self, delta: str) -> list[tuple[bool, str]]:
        if not delta:
            return []
        text = self._carry + delta
        self._carry = ""
        out: list[tuple[bool, str]] = []

        search_text = text.lower() if self._case_insensitive else text

        while text:
            if self._in_think:
                end_tag = self._active_end or "</think>"
                end_search = end_tag.lower() if self._case_insensitive else end_tag
                idx = search_text.find(end_search)
                if idx == -1:
                    chunk, text = self._split_possible_tag_tail(text, end_tag)
                    if chunk:
                        out.append((True, chunk))
                    search_text = text.lower() if self._case_insensitive else text
                    break
                if idx:
                    out.append((True, text[:idx]))
                text = text[idx + len(end_tag) :]
                search_text = text.lower() if self._case_insensitive else text
                self._in_think = False
                self._active_end = None
                continue

            start_idx, start_tag, end_tag = self._find_earliest_start(text)
            if start_idx is None:
                chunk, text = self._split_possible_any_start_tail(text)
                if chunk:
                    out.append((False, chunk))
                search_text = text.lower() if self._case_insensitive else text
                break
            if start_idx:
                out.append((False, text[:start_idx]))
            text = text[start_idx + len(start_tag) :]
            search_text = text.lower() if self._case_insensitive else text
            self._in_think = True
            self._active_end = end_tag

        return out

    def reset(self) -> None:
        self._in_think = False
        self._carry = ""
        self._active_end = None

    def _find_earliest_start(self, text: str) -> tuple[int | None, str, str]:
        search = text.lower() if self._case_insensitive else text
        best_idx: int | None = None
        best_pair: tuple[str, str] | None = None
        for start, end in self._pairs:
            start_search = start.lower() if self._case_insensitive else start
            idx = search.find(start_search)
            if idx == -1:
                continue
            if best_idx is None or idx < best_idx:
                best_idx = idx
                best_pair = (start, end)
        if best_idx is None or best_pair is None:
            return None, "", ""
        return best_idx, best_pair[0], best_pair[1]

    def _split_possible_any_start_tail(self, text: str) -> tuple[str, str]:
        # Keep a possible partial start-tag tail for any known start tag.
        tail = text.lower() if self._case_insensitive else text
        best_keep = 0
        best_tail = ""
        for start, _ in self._pairs:
            max_keep = min(len(start) - 1, len(text))
            start_search = start.lower() if self._case_insensitive else start
            for k in range(max_keep, 0, -1):
                if tail.endswith(start_search[:k]) and k > best_keep:
                    best_keep = k
                    best_tail = text[-k:]
                    break
        if best_keep:
            self._carry = best_tail
            return text[:-best_keep], ""
        return text, ""

    def _split_possible_tag_tail(self, text: str, tag: str) -> tuple[str, str]:
        # Keep a possible partial tag at the end to handle tag splits across deltas.
        max_keep = min(len(tag) - 1, len(text))
        keep = 0
        for k in range(max_keep, 0, -1):
            if text.endswith(tag[:k]):
                keep = k
                break
        if keep:
            self._carry = text[-keep:]
            return text[:-keep], ""
        return text, ""
