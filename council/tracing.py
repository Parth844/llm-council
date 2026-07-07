"""Structured JSON trace logging for debates.

Events stream to traces/{session_id}.jsonl as they happen (crash-safe);
the complete DebateResult is written to traces/{session_id}.json at the end.
"""

from __future__ import annotations

import json
from pathlib import Path

from council.engine import DebateEvent, DebateResult

DEFAULT_TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"


class TraceLogger:
    def __init__(self, traces_dir: Path | str = DEFAULT_TRACES_DIR) -> None:
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def _events_path(self, session_id: str) -> Path:
        return self.traces_dir / f"{session_id}.jsonl"

    def _result_path(self, session_id: str) -> Path:
        return self.traces_dir / f"{session_id}.json"

    def log_event(self, event: DebateEvent) -> None:
        with open(self._events_path(event.session_id), "a") as f:
            f.write(event.model_dump_json() + "\n")

    def save_result(self, result: DebateResult) -> None:
        self._result_path(result.session_id).write_text(
            result.model_dump_json(indent=2)
        )

    def load_session(self, session_id: str) -> dict | None:
        """Full trace: the final result if available, else events so far."""
        result_path = self._result_path(session_id)
        if result_path.exists():
            return json.loads(result_path.read_text())
        events_path = self._events_path(session_id)
        if events_path.exists():
            events = [
                json.loads(line)
                for line in events_path.read_text().splitlines()
                if line.strip()
            ]
            return {"session_id": session_id, "in_progress": True, "events": events}
        return None
