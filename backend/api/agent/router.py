"""
api/agent/router.py — Exposes the POST /agent/ endpoint.
"""
import json
import uuid
from typing import Optional, List
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from api.agent.memory import get_memory
from api.agent.agent import get_agent_executor
from api.limiter import limiter

router = APIRouter()

class AgentRequest(BaseModel):
    query: str = Field(..., description="The user's research question")
    session_id: Optional[str] = Field(None, description="Session ID for memory continuity")

@router.post("/", summary="Invoke the ReAct Research Agent with streaming")
@limiter.limit("5/minute")
async def invoke_agent(request: Request, req: AgentRequest):
    session_id = req.session_id or str(uuid.uuid4())
    memory = get_memory(session_id)
    
    # 1. Load context from Redis/Summary memory
    mem_vars = memory.load_memory_variables({})
    messages = mem_vars.get("chat_history", [])
    
    if not isinstance(messages, list):
        messages = []
        
    human_msg = HumanMessage(content=req.query)
    messages.append(human_msg)
    
    executor = get_agent_executor()
    
    async def event_generator():
        # Yield metadata first so the client gets the session_id
        yield f"data: {json.dumps({'type': 'metadata', 'session_id': session_id})}\n\n"
        
        final_answer = ""
        try:
            async for event in executor.astream_events(
                {"messages": messages},
                version="v2"
            ):
                kind = event["event"]
                name = event["name"]
                lg_node = event.get("metadata", {}).get("langgraph_node", "")

                # ── Stream answer tokens from the agent node only ──────────────
                if kind == "on_chat_model_stream" and lg_node == "agent":
                    chunk = event["data"]["chunk"]

                    # Skip chunks that belong to a tool-call — they are raw JSON fragments
                    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        continue

                    content_str = ""
                    if isinstance(chunk.content, str):
                        content_str = chunk.content
                    elif isinstance(chunk.content, list) and chunk.content:
                        first = chunk.content[0]
                        if isinstance(first, dict):
                            content_str = first.get("text", "")
                        elif isinstance(first, str):
                            content_str = first

                    if content_str:
                        final_answer += content_str
                        yield f"data: {json.dumps({'type': 'answer_chunk', 'content': content_str})}\n\n"

                # ── Emit tool invocations (skip internal graph/LLM nodes) ──────
                elif kind == "on_tool_start" and lg_node not in ("router",):
                    input_data = event["data"].get("input")
                    yield f"data: {json.dumps({'type': 'tool', 'name': name, 'input': input_data})}\n\n"

            # Persist the exchange in session memory
            memory.save_context({"input": req.query}, {"output": final_answer})

            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
