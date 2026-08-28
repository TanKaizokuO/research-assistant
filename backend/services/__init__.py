"""
services package — shared LLM singleton used by all service modules.

Import with:
    from services import LLM
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
KIMI_MODEL = "openai/gpt-oss-20b"

LLM = ChatOpenAI(
    model=KIMI_MODEL,
    base_url=NVIDIA_BASE_URL,
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.1
)
