"""
api/agent/agent.py — Builds and runs the LangGraph agent.
"""
import json
from typing import TypedDict, Annotated, List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from api.dependencies import get_google_key
from api.agent.tools import research_topic, literature_review, citation_graph, ingest_pdf

# Global tools
all_tools = [research_topic, literature_review, citation_graph, ingest_pdf]
tools_by_name = {t.name: t for t in all_tools}

# Maximum number of tool-call rounds before we force the agent to answer.
# 3 rounds is enough for router → tool → synthesis while preventing runaway loops.
_MAX_TOOL_ROUNDS = 3

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    available_tools: List[str]
    tool_calls_count: dict
    tool_rounds: int

class RouterOutput(BaseModel):
    selected_tools: List[str] = Field(description="List of tool names to use for this query.")

async def router_node(state: AgentState):
    """
    Analyzes the query and dynamically injects tools into the state.
    """
    messages = state["messages"]
    last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last_human_msg:
        return {"available_tools": [t.name for t in all_tools]}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=get_google_key(),
        temperature=0.1
    ).with_structured_output(RouterOutput)

    system_prompt = (
        "You are a routing assistant. Decide which tools the agent needs for the user's query.\n"
        "Available tools:\n"
        "- research_topic: For broad topic overviews.\n"
        "- literature_review: For summarizing uploaded papers and formal reviews.\n"
        "- citation_graph: For paper impact, citations, and references.\n"
        "- ingest_pdf: ONLY when user asks to upload or ingest a PDF.\n"
        "Return the list of tool names."
    )

    try:
        decision = await llm.ainvoke([SystemMessage(content=system_prompt), last_human_msg])
        selected = [t for t in decision.selected_tools if t in tools_by_name]
        return {"available_tools": selected if selected else [t.name for t in all_tools]}
    except Exception as e:
        return {"available_tools": [t.name for t in all_tools]}

async def agent_node(state: AgentState):
    """
    The main agent node that decides the next action.
    When the tool-round cap is reached, the LLM is called WITHOUT tools
    so it is forced to produce a final text answer.
    """
    messages = state["messages"]
    available_tools = state.get("available_tools", [t.name for t in all_tools])
    tool_rounds = state.get("tool_rounds") or 0
    
    # Once we've hit the tool-round cap, don't offer tools — force a text answer
    at_cap = tool_rounds >= _MAX_TOOL_ROUNDS
    
    active_tools = [] if at_cap else [
        tools_by_name[name] for name in available_tools if name in tools_by_name
    ]
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=get_google_key(),
        temperature=0.1,
        max_tokens=4096,
    )
    
    if active_tools:
        llm_with_tools = llm.bind_tools(active_tools)
    else:
        llm_with_tools = llm
    
    if at_cap:
        system_msg = SystemMessage(
            content=(
                "You are an expert research assistant. You have already gathered research data from tools.\n"
                "You MUST now synthesize ALL the tool results above into a comprehensive, well-formatted answer.\n"
                "Do NOT attempt to call any tools. Provide your final answer directly."
            )
        )
    else:
        system_msg = SystemMessage(
            content=(
                "You are an expert research assistant. You have access to tools via the function-calling API.\n\n"
                "STRICT RULES:\n"
                "1. ONLY pass the exact parameters each tool expects — never add extra parameters.\n"
                "2. After receiving tool results, ALWAYS synthesise them into a comprehensive, well-formatted answer.\n"
                "3. NEVER output raw JSON in your text response. Use the function-calling mechanism for tool calls.\n"
                "4. If a tool returns an error, explain what happened and try an alternative approach.\n"
                "5. Each tool should only be called ONCE per query unless the first call explicitly failed."
            )
        )
    
    msgs_to_send = [system_msg] + list(messages)
    response = await llm_with_tools.ainvoke(msgs_to_send)
    return {"messages": [response]}

async def tools_node(state: AgentState):
    """
    Executes the tool calls requested by the agent.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_calls_count = dict(state.get("tool_calls_count") or {})
    tool_rounds = (state.get("tool_rounds") or 0) + 1
        
    results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        
        # Deterministic Control 1: PDF Ingestion Human-in-the-loop
        if tool_name == "ingest_pdf":
            results.append(ToolMessage(
                tool_call_id=tool_call["id"], 
                name=tool_name, 
                content="WORKFLOW PAUSED: Please instruct the user to use the UI upload to ingest PDF files, then stop."
            ))
            continue
            
        tool_calls_count[tool_name] = tool_calls_count.get(tool_name, 0) + 1
        
        # Deterministic Control 2: Rate limit per-tool calls (max 1 per tool)
        if tool_calls_count[tool_name] > 1:
            results.append(ToolMessage(
                tool_call_id=tool_call["id"], 
                name=tool_name, 
                content=(
                    f"You already called {tool_name} and received results. "
                    "DO NOT call it again. Synthesize the results you have into a final answer NOW."
                )
            ))
            continue
            
        tool = tools_by_name[tool_name]
        try:
            if hasattr(tool, "ainvoke"):
                result = await tool.ainvoke(tool_call["args"])
            else:
                result = tool.invoke(tool_call["args"])
            results.append(ToolMessage(
                tool_call_id=tool_call["id"], 
                name=tool_name, 
                content=str(result)
            ))
        except Exception as e:
            results.append(ToolMessage(
                tool_call_id=tool_call["id"], 
                name=tool_name, 
                content=f"Error: {str(e)}"
            ))
            
    return {"messages": results, "tool_calls_count": tool_calls_count, "tool_rounds": tool_rounds}

def should_continue(state: AgentState):
    """
    Determine whether to continue to tools or end.
    Hard-caps at _MAX_TOOL_ROUNDS to prevent infinite loops.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Hard cap on total tool rounds
    tool_rounds = state.get("tool_rounds") or 0
    if tool_rounds >= _MAX_TOOL_ROUNDS:
        return "end"
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

# Compile once at import time — avoids re-compiling per request
_executor = None

def get_agent_executor():
    """
    Returns the compiled LangGraph application (singleton).
    """
    global _executor
    if _executor is not None:
        return _executor

    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    
    workflow.set_entry_point("router")
    workflow.add_edge("router", "agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    _executor = workflow.compile()
    return _executor
