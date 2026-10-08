"""
STEP 8 — Human-in-the-loop: pause the graph and wait for a person to decide.

Concept: `interrupt()` pauses a running graph at a specific point and
surfaces a payload describing what it needs. The graph is genuinely
paused (its state is saved via the checkpointer, same one from step 07) —
you can close the process and come back later. To continue, call
`.invoke(Command(resume=<value>), config)` with the same thread_id.

This is how you build agents that ask "should I actually do this?" before
an irreversible action (sending an email, making a purchase, deleting
something) instead of just doing it.

Important LangGraph detail: when a graph resumes after an interrupt, the
node containing the `interrupt()` call RE-RUNS FROM THE TOP. Keep
everything before the `interrupt()` call inside that node idempotent
(safe to repeat) — don't put a side effect like "charge the customer"
before the interrupt in the same node.

Run:
  python 08_human_in_the_loop.py
Then type your answer at the prompt when it pauses.
"""

import os
import uuid
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


@tool
def send_email(to: str, subject: str) -> str:
    """Send an email. This is a sensitive action — always confirm with a human first."""
    return f"Email sent to {to} with subject '{subject}'."


llm = ChatGroq(model=MODEL, temperature=0).bind_tools([send_email])


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_tool_call: dict


def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "human_approval"
    return END


def human_approval_node(state: AgentState):
    last_message = state["messages"][-1]
    call = last_message.tool_calls[0]

    # Pause here. The dict passed to interrupt() is shown to whoever resumes this run.
    decision = interrupt(
        {
            "question": f"The agent wants to call `{call['name']}` with args {call['args']}. Approve? (yes/no)",
        }
    )

    if str(decision).strip().lower() in ("y", "yes"):
        result = send_email.invoke(call["args"])
    else:
        result = "Tool call rejected by human reviewer."

    from langchain_core.messages import ToolMessage

    return {"messages": [ToolMessage(content=result, tool_call_id=call["id"])]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("human_approval", human_approval_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", route_after_agent, {"human_approval": "human_approval", END: END})
graph_builder.add_edge("human_approval", "agent")

graph = graph_builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Send an email to priya@example.com about the Q3 numbers."}]},
        config,
    )

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(payload["question"])
        answer = input("> ")
        result = graph.invoke(Command(resume=answer), config)

    print("\nFinal:", result["messages"][-1].content)
