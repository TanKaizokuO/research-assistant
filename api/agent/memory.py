"""
api/agent/memory.py — Manages per-session conversation memory.
"""
from typing import Dict
from langchain_classic.memory import ConversationBufferWindowMemory

# In-memory dictionary for storing session memories
_sessions: Dict[str, ConversationBufferWindowMemory] = {}

def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    """
    Get or create a memory buffer for a given session ID.
    Returns a ConversationBufferWindowMemory with k=10 to keep the context window manageable.
    """
    if session_id not in _sessions:
        # Create new memory with return_messages=True for ChatModels
        _sessions[session_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,
            return_messages=True
        )
    return _sessions[session_id]

def clear_memory(session_id: str) -> bool:
    """
    Clear the memory for a given session ID.
    Returns True if the session existed and was cleared, False otherwise.
    """
    if session_id in _sessions:
        _sessions[session_id].clear()
        del _sessions[session_id]
        return True
    return False

