"""
STEP 4 — Build the agent loop yourself with LangGraph (no create_agent shortcut).

Concept: `create_agent` in step 03 hid the actual mechanics from you. Under
the hood, an agent is a GRAPH:

    START -> [agent node: ask the LLM what to do next]
                |
                +-- if it wants to call a tool --> [tool node] --> back to [agent node]
                |
                +-- if it has a final answer -----> END

LangGraph makes this explicit:
  - STATE:  a shared dict (here, a growing list of messages) passed between nodes.
  - NODES:  plain Python functions that take the state and return updates to it.
  - EDGES:  wiring between nodes. A "conditional edge" picks the next node
            based on the state (that's the "loop back to tools, or finish?" decision).

This is the foundation for everything more advanced in LangGraph: multi-agent
systems, human-in-the-loop approval steps, retries, parallel branches — all
of it is just more nodes and edges on top of this same skeleton.

Run:
  python 04_langgraph_agent.py
"""

import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


# --- 1. Define the tools (same idea as step 03) ---
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    fake_data = {"hyderabad": "32°C, humid", "bengaluru": "24°C, cloudy"}
    return fake_data.get(city.lower(), f"No data for {city}, assume 28°C and sunny.")


tools = [get_weather]

# `.bind_tools()` tells the model which tools exist and their schemas.
llm = ChatGroq(model=MODEL, temperature=0).bind_tools(tools)


# --- 2. Define the shared state ---
# `add_messages` is a special reducer: instead of overwriting the list on
# every update, it APPENDS new messages to it. This is what makes the
# conversation accumulate across loop iterations.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# --- 3. Define the nodes ---
def agent_node(state: AgentState):
    """Ask the LLM: given the conversation so far, what's next?"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # add_messages appends this to the state


tool_node = ToolNode(tools)  # a ready-made node that executes whichever tool the LLM asked for


# --- 4. Wire the graph ---
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
# `tools_condition` is a prebuilt helper: it checks whether the LLM's last
# message contains tool calls. If yes -> route to "tools". If no -> route to END.
graph_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
graph_builder.add_edge("tools", "agent")  # after running a tool, go back and ask the LLM again

graph = graph_builder.compile()


if __name__ == "__main__":
    # Optional: print an ASCII view of the graph structure you just built.
    print(graph.get_graph().draw_ascii())
    print()

    result = graph.invoke({"messages": [{"role": "user", "content": "What's the weather in Bengaluru?"}]})

    for m in result["messages"]:
        role = m.type
        content = m.content if m.content else getattr(m, "tool_calls", "")
        print(f"[{role}] {content}")
