"""
STEP 3 — Give the model tools and let it decide when to use them (an "agent").

Concept: A plain chain always runs the same steps. An AGENT is a chain
where the model itself decides, at each turn, whether to call a tool
(a plain Python function) or answer directly, then loops until it has a
final answer. LangChain's `create_agent` builds this loop for you using a
ReAct-style pattern under the hood (which is itself implemented in
LangGraph — more on that in step 04).

Groq's Llama models are solid at tool calling, so this should work
reliably out of the box.

Run:
  python 03_tool_agent.py
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

llm = ChatGroq(model=MODEL, temperature=0)


# A "tool" is just a normal Python function. The docstring and type hints
# are NOT decoration — LangChain reads them and hands them to the model so
# it knows the tool exists and how to call it. Keep docstrings precise.
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # Faked here on purpose — swap in a real API call (e.g. Open-Meteo, which is free
    # and needs no key) once you're comfortable with the pattern.
    fake_data = {"hyderabad": "32°C, humid", "bengaluru": "24°C, cloudy"}
    return fake_data.get(city.lower(), f"No data for {city}, assume 28°C and sunny.")


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    return a + b


agent = create_agent(
    model=llm,
    tools=[get_weather, add_numbers],
    system_prompt=(
        "You are a helpful assistant. Use the available tools when they "
        "would give a more accurate answer than guessing."
    ),
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Hyderabad, and what's 42 + 58?"}]}
    )

    # `create_agent` returns the full conversation state — the last message is the answer.
    final_message = result["messages"][-1]
    print("Final answer:", final_message.content)

    print("\n--- Full trace of messages (see how the agent called tools) ---")
    for m in result["messages"]:
        print(f"[{m.type}] {m.content if m.content else m.tool_calls}")
