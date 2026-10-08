"""
STEP 11 — Structured state: track more than just messages.

Concept: every graph so far only tracked a growing `messages` list. Real
graphs usually need to track other things too — a retry counter, whether
something was approved, a running total, extracted fields. You can add
any field to your state `TypedDict` (or a Pydantic model, shown below);
each node just returns a dict with whichever keys it wants to update.

This example tracks a `retry_count` and stops asking the model to retry
after a max, plus uses Pydantic for validation instead of a plain
TypedDict — useful once your state has fields that must satisfy some
constraint (e.g. a score between 0 and 1).

Run:
  python 11_structured_state.py
"""

import os
from dotenv import load_dotenv
from typing import Annotated, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

llm = ChatGroq(model=MODEL, temperature=0.7)  # higher temperature so retries can plausibly differ

MAX_RETRIES = 3


# A Pydantic model instead of a TypedDict — LangGraph supports both.
# Pydantic gives you validation (e.g. ge=0 below) and clearer errors if a
# node returns a bad value.
class AgentState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    retry_count: int = 0
    approved: bool = False


def draft_node(state: AgentState):
    """Ask the model for a one-line product tagline; keep asking until it's short enough."""
    response = llm.invoke(
        state.messages
        + [{"role": "user", "content": "Give me ONE tagline, under 8 words, for an AI observability tool. No quotes, no punctuation at the end."}]
    )
    return {"messages": [response], "retry_count": state.retry_count + 1}


def check_node(state: AgentState) -> AgentState:
    last = state.messages[-1].content
    word_count = len(last.split())
    approved = word_count <= 8
    print(f"Attempt {state.retry_count}: \"{last}\" ({word_count} words) -> {'OK' if approved else 'too long, retrying'}")
    return {"approved": approved}


def route(state: AgentState) -> str:
    if state.approved:
        return "end"
    if state.retry_count >= MAX_RETRIES:
        print(f"Gave up after {MAX_RETRIES} attempts.")
        return "end"
    return "retry"


graph_builder = StateGraph(AgentState)
graph_builder.add_node("draft", draft_node)
graph_builder.add_node("check", check_node)
graph_builder.add_edge(START, "draft")
graph_builder.add_edge("draft", "check")
graph_builder.add_conditional_edges("check", route, {"retry": "draft", "end": END})
graph = graph_builder.compile()


if __name__ == "__main__":
    result = graph.invoke({"messages": []})
    print("\nFinal tagline:", result["messages"][-1].content)
    print("Total attempts:", result["retry_count"])
