"""
api/dependencies.py — FastAPI dependency functions for shared secrets.

API keys are read once from the environment (via .env) and cached with
lru_cache so the env-var lookup only happens once per process lifetime.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache()
def get_tavily_key() -> str:
    """Return the Tavily API key, raising RuntimeError if not set."""
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not set in the environment")
    return key


@lru_cache()
def get_nvidia_key() -> str:
    """Return the NVIDIA API key, raising RuntimeError if not set."""
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is not set in the environment")
    return key
