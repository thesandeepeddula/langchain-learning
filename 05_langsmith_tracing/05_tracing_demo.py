"""
STEP 5 — Turn on LangSmith tracing to SEE what your agent is doing.

Concept: LangSmith is an observability tool. It doesn't change how your
code runs — it just records every step (prompts sent, tokens used, tool
calls, latency, errors) so you can inspect them in a web UI instead of
guessing from print statements. This becomes essential once agents get
more than 2-3 steps.

It's a hosted product, but the free "Developer" plan gives you a generous
monthly trace allowance with a short retention window — more than enough
for learning, and no credit card required to sign up. (Check current
numbers at https://smith.langchain.com/pricing since free-tier limits do
shift over time.)

Setup (one-time):
  1. Go to https://smith.langchain.com and create a free account.
  2. Settings -> API Keys -> Create API Key.
  3. Copy .env.example to .env (in the project root) and paste your key
     into LANGSMITH_API_KEY. Set LANGSMITH_TRACING=true.

That's it — you do NOT need to change any of your LangChain/LangGraph code.
LangSmith hooks in automatically by reading those environment variables.
This script just re-runs step 04's graph so you have something to look at.

Run:
  python 05_tracing_demo.py
Then open https://smith.langchain.com and check the "langchain-learning"
project — you'll see a full trace tree of this run: the graph nodes, the
LLM call, the tool call, tokens, and latency per step.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LANGSMITH_TRACING", "false").lower() != "true":
    print(
        "LANGSMITH_TRACING is not set to true in your .env file.\n"
        "This script will still run, but nothing will be sent to LangSmith.\n"
        "See the setup steps in this file's docstring.\n"
    )

# Make step 04's graph importable from here without turning the project into
# a package — simplest option for a learning repo.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "04_langgraph_agent"))
from importlib.machinery import SourceFileLoader  # noqa: E402

mod = SourceFileLoader(
    "graph_module", os.path.join(os.path.dirname(__file__), "..", "04_langgraph_agent", "04_langgraph_agent.py")
).load_module()

if __name__ == "__main__":
    result = mod.graph.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Hyderabad? Also, who are you?"}]}
    )
    print(result["messages"][-1].content)
    # The web dashboard host matches the API host minus the "api." prefix
    # (e.g. apac.api.smith.langchain.com -> apac.smith.langchain.com), since
    # LangSmith's API and dashboard live on region-specific subdomains.
    api_host = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    dashboard_host = api_host.replace("api.smith.langchain.com", "smith.langchain.com").rstrip("/")
    print(
        f"\nIf tracing was on, open {dashboard_host} -> your project "
        f"'{os.getenv('LANGSMITH_PROJECT', 'langchain-learning')}' to see the full run."
    )
