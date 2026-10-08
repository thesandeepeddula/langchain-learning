"""
STEP 9 — Streaming: watch the graph work instead of waiting for the end.

Concept: `.invoke()` blocks until the entire run finishes. `.stream()`
yields output incrementally, and LangGraph supports a few different
"stream modes" depending on what granularity you want:

  - "updates": yields after each NODE finishes — good for watching an
    agent's steps happen live (tool called, then agent responded, etc).
  - "messages": yields individual LLM TOKENS as they're generated — good
    for a chat UI that types responses out character by character.

Everything else about the graph (nodes, edges, state) is identical to
step 04 — streaming is purely about how you consume the output.

Run:
  python 09_streaming.py
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


def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    fake_data = {"hyderabad": "32°C, humid", "bengaluru": "24°C, cloudy"}
    return fake_data.get(city.lower(), f"No data for {city}, assume 28°C and sunny.")


tools = [get_weather]
llm = ChatGroq(model=MODEL, temperature=0).bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()


if __name__ == "__main__":
    question = {"messages": [{"role": "user", "content": "What's the weather in Hyderabad?"}]}

    print("=== stream_mode='updates' (one event per node) ===")
    for event in graph.stream(question, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n[node: {node_name}] ->")
            for m in node_output["messages"]:
                print(" ", m.content if m.content else getattr(m, "tool_calls", ""))

    print("\n\n=== stream_mode='messages' (token-by-token from the LLM) ===")
    for chunk, metadata in graph.stream(question, stream_mode="messages"):
        # `chunk` is a partial AIMessageChunk; print its text as it arrives, no newline, no buffering.
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()
