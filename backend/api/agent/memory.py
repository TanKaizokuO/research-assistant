"""
api/agent/memory.py — Manages per-session conversation memory.

ConversationSummaryBufferMemory was removed in langchain v1.x.
We implement a lightweight shim that exposes the same two methods
the router depends on:
  - load_memory_variables({}) -> {"chat_history": [BaseMessage, ...]}
  - save_context({"input": ...}, {"output": ...})

Redis is used for durable message history. If Redis is unavailable the
session falls back to an in-process InMemoryChatMessageHistory.
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

try:
    from langchain_community.chat_message_histories import RedisChatMessageHistory
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

# Keep instances alive for the lifetime of the worker process
_sessions: Dict[str, "MemoryShim"] = {}

_REDIS_URL = "redis://localhost:6379/0"
_MAX_MESSAGES = 20  # keep last N messages to bound context size


class MemoryShim:
    """
    Minimal drop-in replacement for ConversationSummaryBufferMemory.

    Stores raw messages in Redis (or in-process fallback) and returns
    the last `max_messages` as the chat history.
    """

    def __init__(self, session_id: str, max_messages: int = _MAX_MESSAGES):
        self.session_id = session_id
        self.max_messages = max_messages

        if _REDIS_AVAILABLE:
            try:
                self._history = RedisChatMessageHistory(session_id, url=_REDIS_URL)
                # Probe connection
                _ = self._history.messages
            except Exception:
                self._history = InMemoryChatMessageHistory()
        else:
            self._history = InMemoryChatMessageHistory()

    # ------------------------------------------------------------------
    # Public API (mirrors ConversationSummaryBufferMemory)
    # ------------------------------------------------------------------

    def load_memory_variables(self, _inputs: Dict[str, Any]) -> Dict[str, List[BaseMessage]]:
        """Return the last N messages as ``{"chat_history": [...]}``."""
        msgs: List[BaseMessage] = self._history.messages
        return {"chat_history": msgs[-self.max_messages:]}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Persist a human/AI exchange to history."""
        human_text = inputs.get("input", "")
        ai_text = outputs.get("output", "")

        if human_text:
            self._history.add_message(HumanMessage(content=human_text))
        if ai_text:
            self._history.add_message(AIMessage(content=ai_text))

    def clear(self) -> None:
        self._history.clear()


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def get_memory(session_id: str) -> MemoryShim:
    """Get or create a MemoryShim for *session_id*."""
    if session_id not in _sessions:
        _sessions[session_id] = MemoryShim(session_id)
    return _sessions[session_id]


def clear_memory(session_id: str) -> bool:
    """Clear the stored history for *session_id*."""
    shim = _sessions.pop(session_id, None)
    if shim is not None:
        shim.clear()
        return True

    # Session not in local cache — still try to clear Redis
    if _REDIS_AVAILABLE:
        try:
            RedisChatMessageHistory(session_id, url=_REDIS_URL).clear()
        except Exception:
            pass

    return True
