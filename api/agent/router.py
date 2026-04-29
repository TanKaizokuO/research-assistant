"""
api/agent/router.py — Exposes the POST /agent/ endpoint.
"""
import uuid
from typing import Optional, List
from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.agent.memory import get_memory
from api.agent.agent import get_agent_executor

router = APIRouter()

class AgentRequest(BaseModel):
    query: str = Field(..., description="The user's research question")
    session_id: Optional[str] = Field(None, description="Session ID for memory continuity")

class AgentResponse(BaseModel):
    answer: str = Field(..., description="The agent's final synthesized response")
    session_id: str = Field(..., description="Session ID to echo back for next turn")
    steps: List[str] = Field(default_factory=list, description="Tool calls made")

@router.post("/", response_model=AgentResponse, summary="Invoke the ReAct Research Agent")
async def invoke_agent(req: AgentRequest):
    # 1. Generate session_id if not present
    session_id = req.session_id or str(uuid.uuid4())
    
    # 2. Retrieve memory and executor
    session_memory = get_memory(session_id)
    executor = get_agent_executor(session_memory)
    
    # 3. Invoke the agent
    # We use ainvoke for asynchronous execution
    try:
        result = await executor.ainvoke({"input": req.query})
        
        answer = result.get("output", "I could not generate an answer.")
        
        # Extract intermediate steps if they exist in the result
        # intermediate_steps is a list of tuples: (AgentAction, observation)
        steps_info = []
        if "intermediate_steps" in result:
            for action, observation in result["intermediate_steps"]:
                steps_info.append(f"Tool {action.tool} with input '{action.tool_input}'")
                
        return AgentResponse(
            answer=answer,
            session_id=session_id,
            steps=steps_info
        )
    except Exception as e:
        return AgentResponse(
            answer=f"An error occurred while processing your request: {str(e)}",
            session_id=session_id,
            steps=[]
        )

