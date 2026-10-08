"""
STEP 10 — Multi-agent: a supervisor that routes between two specialists.

Concept: instead of one agent with every tool, split responsibilities:
each sub-agent gets its own focused tools and system prompt, and a
SUPERVISOR node decides which one should handle the current request. The
supervisor is itself just another LLM call with structured output (it
picks one of a fixed set of names) — same building blocks as before, just
one more layer of routing.

This example is deliberately shaped like a real internal-tooling use
case: a "metrics" agent that looks up usage numbers, and a "writer" agent
that turns raw numbers into a short prose summary. A supervisor decides,
per user message, which one should act. This is the shape you'd extend
into something like an AICT usage-Q&A agent.

Run:
  python 10_multi_agent.py
"""

import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

llm = ChatGroq(model=MODEL, temperature=0)


# --- Specialist 1: looks up numbers (fake data, swap for a real query later) ---
def get_usage_metrics(product: str) -> str:
    """Look up adoption/usage metrics for a given internal product name."""
    fake = {
        "aict": "1,240 active users this month, up 18% month-over-month; 340 workflows onboarded.",
        "cmdb": "8,900 active users this month, up 2% month-over-month; stable.",
    }
    return fake.get(product.lower(), f"No usage data found for '{product}'.")


metrics_agent_llm = llm.bind_tools([get_usage_metrics])


def metrics_agent(state: "AgentState"):
    system = {"role": "system", "content": "You look up usage metrics using the get_usage_metrics tool, then report the raw numbers plainly."}
    response = metrics_agent_llm.invoke([system] + state["messages"])
    return {"messages": [response]}


# --- Specialist 2: turns numbers into a short narrative, no tools needed ---
def writer_agent(state: "AgentState"):
    system = {
        "role": "system",
        "content": "You write a short (2-3 sentence) executive-friendly summary of whatever numbers appear earlier in the conversation. Don't invent numbers not already present.",
    }
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


# --- Supervisor: decides which specialist handles this turn ---
class RouteDecision(BaseModel):
    next: Literal["metrics_agent", "writer_agent", "END"]


router_llm = llm.with_structured_output(RouteDecision)


def supervisor(state: "AgentState"):
    system = {
        "role": "system",
        "content": (
            "Decide who should act next. Route to 'metrics_agent' if the user is asking for "
            "usage/adoption numbers that haven't been looked up yet. Route to 'writer_agent' if "
            "numbers are already in the conversation and need summarizing into prose. Route to "
            "'END' if the last message already fully answers the user."
        ),
    }
    decision = router_llm.invoke([system] + state["messages"])
    return {"next": decision.next}


def route(state: "AgentState") -> str:
    return state["next"]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


graph_builder = StateGraph(AgentState)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("metrics_agent", metrics_agent)
graph_builder.add_node("metrics_tools", ToolNode([get_usage_metrics]))
graph_builder.add_node("writer_agent", writer_agent)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor", route, {"metrics_agent": "metrics_agent", "writer_agent": "writer_agent", "END": END}
)
# metrics_agent may need to actually call its tool before it has an answer.
graph_builder.add_conditional_edges("metrics_agent", tools_condition, {"tools": "metrics_tools", "__end__": "supervisor"})
graph_builder.add_edge("metrics_tools", "metrics_agent")
# writer_agent never calls tools, so it always goes straight back to the supervisor.
graph_builder.add_edge("writer_agent", "supervisor")

graph = graph_builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "How is AICT adoption trending, summarized for my manager?"}]},
        {"recursion_limit": 15},  # multi-agent loops can run longer than a single agent; raise the default limit
    )

    print("--- Full run ---")
    for m in result["messages"]:
        if isinstance(m, AIMessage) and not m.content:
            continue  # skip empty tool-call-only messages for readability
        print(f"[{m.type}] {m.content}")
