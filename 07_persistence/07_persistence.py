"""
STEP 7 — Persistence: give the graph memory across separate calls.

Concept: Every script so far ran `.invoke()` once and forgot everything.
A CHECKPOINTER saves the graph's state after every node runs, keyed by a
`thread_id`. Call `.invoke()` again with the SAME thread_id and the graph
picks up where it left off — that's how you get multi-turn conversations
without manually re-sending the whole history yourself.

This is the single biggest upgrade from step 04: same graph, same nodes,
same edges — just add a checkpointer and a thread_id.

`InMemorySaver` (used here) keeps state in RAM only — it's gone when the
script exits. That's fine for learning. For anything that needs to survive
a restart, swap in `SqliteSaver` (see the commented-out lines below) — a
single local file, still free, still no server to run.

Run:
  python 07_persistence.py
"""

import os
import uuid
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# For durable (file-based) persistence instead, use this and pass a real path:
#   from langgraph.checkpoint.sqlite import SqliteSaver
#   with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
#       ... build graph, run it, all inside this `with` block ...
from langchain_groq import ChatGroq

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

llm = ChatGroq(model=MODEL, temperature=0)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # thread_id is the "conversation ID" — everything under the same ID shares state.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("--- Turn 1 ---")
    result1 = graph.invoke({"messages": [{"role": "user", "content": "My name is Sandy."}]}, config)
    print(result1["messages"][-1].content)

    print("\n--- Turn 2 (new invoke() call, same thread_id) ---")
    result2 = graph.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
    print(result2["messages"][-1].content)  # the model remembers, because state persisted

    print("\n--- Turn 3, but with a DIFFERENT thread_id (fresh conversation) ---")
    other_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result3 = graph.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, other_config)
    print(result3["messages"][-1].content)  # no memory — different thread, clean slate

    print("\n--- Full saved state for thread 1 ---")
    snapshot = graph.get_state(config)
    for m in snapshot.values["messages"]:
        print(f"[{m.type}] {m.content}")
