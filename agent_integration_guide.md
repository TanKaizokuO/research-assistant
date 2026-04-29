# Integrating LangChain ReAct Agents into Your Research Assistant FastAPI Backend

## Overview

This guide walks you through converting your existing FastAPI backend into an **agent-driven system**. Instead of the client deciding which endpoint to call, a LangChain ReAct agent with memory and state autonomously decomposes a user's research query and routes it across your existing pipelines.

Your existing services (`/research/`, `/literature/`, `/citations/`) remain unchanged. The agent sits on top of them as an orchestration layer.

---

## How the Agent Works

The ReAct (Reasoning + Acting) pattern works in a loop:

1. **Thought** — The LLM reasons about what it needs to do next
2. **Action** — It picks a tool and calls it with arguments
3. **Observation** — It reads the tool's output
4. **Repeat** — Until it has enough to produce a final answer

For your backend, the tools are your three existing pipelines. The agent decides which ones to call, in what order, based on the query.

```
User: "Give me a deep review of transformer architectures with key citations"
    │
    ▼
Agent Thought: "I need to search for papers, review literature, and find citations."
    │
    ├── Action: research_topic("transformer architectures")
    ├── Action: literature_review("transformer architectures")
    └── Action: citation_graph("Attention Is All You Need")
    │
    ▼
Agent Final Answer: Synthesized response combining all three results
```

---

## Proposed File Structure

Add the following files alongside your existing structure. Do **not** modify existing service files.

```
api/
├── app.py                     # existing — add one new router here
├── services/
│   ├── research.py            # existing — no changes
│   ├── literature.py          # existing — no changes
│   └── citations.py           # existing — no changes
└── agent/
    ├── __init__.py
    ├── tools.py               # wraps your services as LangChain Tools
    ├── memory.py              # manages per-session conversation memory
    ├── agent.py               # builds and runs the ReAct agent
    └── router.py              # FastAPI router exposing /agent/ endpoint
```

---

## Step 1 — Wrap Services as LangChain Tools (`api/agent/tools.py`)

LangChain Tools are Python functions decorated with `@tool`. The agent sees the **docstring** as its instructions for when to use each tool, so write clear, specific descriptions.

**Key rules for tool docstrings:**
- Describe exactly what the tool returns, not just what it does
- Specify what input format it expects (a topic string, a paper title, a DOI, etc.)
- Tell the agent when to prefer this tool over others

**One tool per existing pipeline:**

| Tool Name | Wraps | When the agent should use it |
|---|---|---|
| `research_topic` | `/research/` service | Broad topic overviews, recent findings |
| `literature_review` | `/literature/review` service | Structured review from ingested PDFs |
| `citation_graph` | `/citations/` service | Understanding a specific paper's impact |
| `ingest_pdf` | `/literature/ingest` service | Only when user explicitly uploads a file |

**Important:** Tools should call your existing service functions directly — do not duplicate logic. If your services are currently only called from FastAPI route handlers, refactor the core logic into importable functions first, then have both the route handler and the tool call that shared function.

---

## Step 2 — Add Session Memory (`api/agent/memory.py`)

Memory lets the agent remember context across multiple turns in the same session. Without it, every message is stateless.

**What to implement:**

- A dictionary (or Redis if you want persistence across restarts) keyed by `session_id`
- Each session stores a `ConversationBufferWindowMemory` (keeps the last N turns to avoid hitting the context limit)
- A function to get-or-create memory for a given session ID
- A function to clear a session's memory

**LangChain memory class to use:** `ConversationBufferWindowMemory` with `return_messages=True` and a window of 10–15 turns. This prevents the context window from growing unbounded in long sessions.

**Session ID strategy:** Accept a `session_id` string from the client in the request body. If none is provided, generate a UUID and return it in the response so the client can send it back on the next turn.

---

## Step 3 — Build the ReAct Agent (`api/agent/agent.py`)

This is the core of the integration. You will:

1. **Initialize the LLM** — Use the same NVIDIA LLM you already use in your services, configured via `langchain_nvidia_ai_endpoints`
2. **Register tools** — Pass the list of tool functions from `tools.py`
3. **Create the agent** — Use `create_react_agent` with a prompt that includes memory placeholders
4. **Wrap in AgentExecutor** — Set `verbose=True` during development to see the reasoning trace, `handle_parsing_errors=True` to avoid crashes on malformed LLM output
5. **Inject memory** — Pass the session's memory object to the executor

**Prompt structure for ReAct with memory:**

Your system prompt should contain four components in this order:
- Role and context ("You are a research assistant with access to the following tools...")
- Tool descriptions (injected automatically by LangChain via the `{tools}` placeholder)
- Conversation history (injected from memory via `{chat_history}`)
- Scratchpad for reasoning (injected via `{agent_scratchpad}`)

**Agent executor settings to configure:**

| Setting | Recommended value | Reason |
|---|---|---|
| `max_iterations` | 6–8 | Prevents infinite loops on ambiguous queries |
| `max_execution_time` | 120 seconds | Guards against slow external API calls |
| `handle_parsing_errors` | `True` | LLMs sometimes produce malformed ReAct output |
| `verbose` | `True` in dev, `False` in prod | Useful for debugging the reasoning trace |

---

## Step 4 — Expose via FastAPI Router (`api/agent/router.py`)

Create a new router with a single `POST /agent/` endpoint.

**Request body should include:**
- `query: str` — the user's research question
- `session_id: Optional[str]` — for memory continuity; generate one if absent

**Response body should include:**
- `answer: str` — the agent's final synthesized response
- `session_id: str` — echo back so the client can use it next turn
- `steps: List[str]` — optional, the tool calls made (useful for transparency in a UI)

**In `api/app.py`**, include the new router:
```
app.include_router(agent_router, prefix="/agent", tags=["agent"])
```

No other changes to `app.py` are needed.

---

## Step 5 — Handle Streaming (Optional but Recommended)

Research tasks are slow. Users should see output progressively rather than waiting 30–60 seconds for a complete response.

**How to implement:**
- Use FastAPI's `StreamingResponse` with `media_type="text/event-stream"`
- Use LangChain's `AsyncIteratorCallbackHandler` to stream tokens as they are generated
- Stream both the reasoning steps (tool calls) and the final answer tokens
- The client reads the event stream and appends chunks to the UI

This is especially important for your use case because each tool call (arXiv, Semantic Scholar, Tavily) involves real network requests that add latency.

---

## Step 6 — Add a New Dependency (`requirements.txt`)

Add the following to your existing `requirements.txt`:

```
langchain
langchain-nvidia-ai-endpoints
langgraph                     # optional, only if you later want multi-agent graphs
```

You already have LangChain if you use it for your LLM calls. Check whether `langchain-core` is already listed and avoid duplicate installs.

---

## Autonomous Task Decomposition — How It Works in Practice

The agent's ability to decompose tasks comes entirely from the tool docstrings and the system prompt. You do not need to write a router or classifier. The LLM does this reasoning natively.

**Example decompositions:**

| User Query | Tools the agent should invoke |
|---|---|
| "What are the latest advances in RAG?" | `research_topic` |
| "Summarize the papers I uploaded about LLMs" | `literature_review` |
| "Who cited the BERT paper and what did they build on?" | `citation_graph` |
| "Give me a full briefing on diffusion models" | `research_topic` → `literature_review` (if PDFs exist) → `citation_graph` |

The agent chooses this routing on its own. If a query is ambiguous, it will use the `research_topic` tool first to gather context before deciding whether to invoke further tools.

---

## Things to Be Careful About

**Tool errors should not crash the agent.** Wrap each tool's internal logic in try/except and return an error string rather than raising an exception. The agent will read the error as an observation and either retry or explain to the user what failed.

**Avoid duplicate context.** If `research_topic` and `literature_review` both call Semantic Scholar, the agent's final answer may repeat the same paper twice. Consider deduplicating results in a shared utility before passing to the LLM.

**Rate limits.** Each tool invocation triggers real API calls (Tavily, arXiv, Semantic Scholar). A single agent run that calls three tools will make significantly more downstream requests than a single endpoint call. Add per-session rate limiting to `/agent/` to protect your API quotas.

**Memory size.** `ConversationBufferWindowMemory` with a window of 10 turns is usually sufficient. If sessions grow very long (multi-hour research sessions), consider summarizing older turns using `ConversationSummaryMemory` instead.

**Observability.** During development, log the full agent trace (every thought, action, and observation) to a file. This is the primary debugging tool when the agent makes wrong routing decisions.

---

## Testing the Agent

Test in this order before exposing to users:

1. **Single-tool queries** — Verify each tool is called correctly in isolation
2. **Multi-tool queries** — Verify the agent sequences tools correctly
3. **Multi-turn queries** — Verify memory persists across turns in the same session
4. **Error cases** — Verify the agent handles tool failures gracefully
5. **Edge cases** — Queries unrelated to research (the agent should say it cannot help, not hallucinate a tool call)

A good test harness runs each case programmatically and checks that the `steps` list in the response contains the expected tool names.

---

## Summary of Changes Required

| File | Action |
|---|---|
| `api/agent/tools.py` | Create — wraps existing services as `@tool` functions |
| `api/agent/memory.py` | Create — manages per-session `ConversationBufferWindowMemory` |
| `api/agent/agent.py` | Create — builds `AgentExecutor` with tools and memory |
| `api/agent/router.py` | Create — exposes `POST /agent/` |
| `api/app.py` | Edit — add one `include_router` line |
| `requirements.txt` | Edit — add `langchain`, `langchain-nvidia-ai-endpoints` |
| Existing service files | **No changes** — agent calls them, not the other way around |
