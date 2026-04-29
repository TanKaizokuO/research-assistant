"""
api/agent/agent.py — Builds and runs the LangChain ReAct agent.
"""
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from api.dependencies import get_nvidia_key
from api.agent.tools import research_topic, literature_review, citation_graph, ingest_pdf

# List of tools to provide to the agent
tools = [research_topic, literature_review, citation_graph, ingest_pdf]

# System prompt for ReAct Agent with Memory
agent_prompt = PromptTemplate.from_template(
    """You are a helpful, intelligent research assistant with access to various research tools.

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""
)

def get_agent_executor(session_memory) -> AgentExecutor:
    """
    Creates and returns an AgentExecutor instance using the tools and provided memory.
    """
    # Initialize the LLM
    nvidia_key = get_nvidia_key()
    llm = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct", # Typical good default model for NVIDIA
        nvidia_api_key=nvidia_key,
        temperature=0.1,
        max_tokens=2048,
    )
    
    # Create the ReAct agent
    agent = create_react_agent(llm, tools, agent_prompt)
    
    # Wrap in an AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=session_memory,
        verbose=True, # Set to True for development visibility
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=6,
        max_execution_time=120
    )
    
    return agent_executor

