"""
services package — shared LLM singleton used by all service modules.

Import with:
    from services import LLM
"""
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

LLM = ChatNVIDIA(model="moonshotai/kimi-k2-instruct-0905")
