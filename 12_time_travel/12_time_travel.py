"""
STEP 12 — Time travel: rewind to an earlier point in a run and branch from there.

Concept: because a checkpointer (step 07) saves state after every node,
LangGraph keeps a full history of checkpoints for a thread. `get_state_history()`
lists them all, oldest to newest. You can pick any earlier checkpoint,
optionally edit its state, and resume execution FROM there — creating a
new branch of history without re-running everything from scratch.

This is genuinely useful for debugging ("what did the state look like
right before it made that weird tool call?") and for building things like
"regenerate this response" buttons that don't require replaying the whole
conversation.

Run:
  python 12_time_travel.py
"""

import os
import uuid
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

llm = ChatGroq(model=MODEL, temperature=0.9)  # high temperature: re-running should give a visibly different answer


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", END)
graph = graph_builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    graph.invoke({"messages": [{"role": "user", "content": "Name one good movie."}]}, config)
    graph.invoke({"messages": [{"role": "user", "content": "Now name one good book."}]}, config)

    print("--- Checkpoint history for this thread (newest first) ---")
    history = list(graph.get_state_history(config))
    for i, snapshot in enumerate(history):
        last_msg = snapshot.values["messages"][-1] if snapshot.values.get("messages") else None
        preview = (last_msg.content[:50] + "...") if last_msg and last_msg.content else "(no message yet)"
        print(f"[{i}] checkpoint_id={snapshot.config['configurable']['checkpoint_id'][:8]}... last message: {preview}")

    # Rewind to right after the FIRST answer (the movie one), before the book question ever happened.
    rewind_target = history[-2]  # history[-1] is the very first checkpoint (empty state); [-2] is just after turn 1
    print(f"\nRewinding to checkpoint after turn 1: {rewind_target.config['configurable']['checkpoint_id'][:8]}...")

    # Resume FROM that checkpoint with a new message — this branches history instead of continuing turn 2's branch.
    branched_result = graph.invoke(
        {"messages": [{"role": "user", "content": "Now name one good song instead."}]},
        rewind_target.config,
    )
    print("\nBranched result:", branched_result["messages"][-1].content)
    print(
        "\nNote: the 'good book' answer from the original run above still exists in history — "
        "you just created a second branch alongside it, you didn't erase it."
    )
