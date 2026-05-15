"""
core/session.py

In-memory session container.  No data is persisted to disk.
end_session() wipes all data — called unconditionally in cli/main.py's
finally block so the contract holds even on KeyboardInterrupt.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


class Session:
    """
    Lightweight in-memory session.

    Stores per-interaction data (transcripts, detected languages, translations)
    as plain dict values.  Everything is cleared when end_session() is called.
    """

    def __init__(self) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.started_at: datetime = datetime.now(tz=timezone.utc)
        self._data: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Store a value in the session."""
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the session."""
        return self._data.get(key, default)

    def append(self, key: str, value: Any) -> None:
        """Append *value* to a list stored at *key* (creates list if absent)."""
        if key not in self._data:
            self._data[key] = []
        self._data[key].append(value)

    def keys(self) -> list[str]:
        return list(self._data.keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def end_session(self) -> None:
        """Wipe all session data from memory.  Idempotent."""
        self._data.clear()
        self.session_id = ""

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Session(id={self.session_id!r}, "
            f"started={self.started_at.isoformat()}, "
            f"keys={self.keys()})"
        )

    def __bool__(self) -> bool:
        """True while session is active (has a session_id)."""
        return bool(self.session_id)
