# LangChain + LangGraph + LangSmith — Zero-Cost Learning Path (Groq API)

Thirteen small scripts, run in order, each one adding exactly one new
concept. The chat model runs through Groq's free API — no local model to
install, so this works regardless of your Mac's specs or macOS version. No
paid plan required anywhere except step 13, which needs LangSmith's free
tier (see that step's docstring).

**Part 1 — LangChain and LangGraph fundamentals:**

```
01_llm_basics/          -> call an LLM through LangChain
02_prompts_chains/      -> prompt templates + LCEL chains (the `|` pipe)
03_tools_agent/         -> create_agent: model decides when to call tools
04_langgraph_agent/     -> build that same agent loop yourself with LangGraph
05_langsmith_tracing/   -> turn on tracing to see the agent's steps in a UI
06_rag_agent/           -> capstone: retrieval (RAG) wired in as a tool
```

**Part 2 — going deeper on LangGraph specifically:**

```
07_persistence/         -> checkpointers: memory across separate invoke() calls
08_human_in_the_loop/   -> interrupt() a run to ask a human before acting
09_streaming/           -> .stream() instead of .invoke() — watch it work live
10_multi_agent/         -> a supervisor routing between two specialist agents
11_structured_state/    -> state beyond a message list (Pydantic, retry counters)
12_time_travel/         -> rewind to an earlier checkpoint and branch from it
13_langsmith_eval/      -> capstone: score your agent against a test dataset
```

> **Note on Ollama:** this project originally used Ollama for a fully
> offline setup, but Ollama's desktop app requires macOS 14+, and this
> Mac is on macOS 12.7.6 — so everything below uses Groq's free API for
> chat instead. Nothing here needs Ollama. If you ever upgrade macOS, you
> can swap `ChatGroq(...)` back for `ChatOllama(...)` in each script and
> nothing else changes — that's the whole point of LangChain's common
> model interface (see step 01's docstring).

## Why this order

LangChain, LangGraph, and LangSmith solve three different problems and
it's easy to blur them together as a beginner:

- **LangChain** — building blocks: models, prompts, chains, tools, agents.
- **LangGraph** — the engine that actually runs an agent's loop (LangChain's
  `create_agent` is built on top of it). You use LangChain's shortcut first
  (step 3), then open the hood and build the same thing by hand (step 4) so
  the abstraction never feels like magic.
- **LangSmith** — a window into what steps 1–4 are actually doing at
  runtime. It's not a coding library at all, just environment variables
  that turn on logging to a web dashboard.

## One-time setup

### 1. Create a free Groq account

1. Go to https://console.groq.com and sign up (no card required).
2. Left sidebar -> API Keys -> Create API Key. Copy it somewhere safe —
   you can't view it again after this screen.

Groq's free tier gives you a monthly allowance of requests/tokens across
models, with `openai/gpt-oss-20b` having the most headroom and
`llama-3.3-70b-versatile` rate-limited to a handful of requests/minute
(fine for occasional use, not for tight loops). Exact numbers shift over
time — check https://console.groq.com/settings/limits once you're signed
in to see your account's current limits.

### 2. Python environment

```bash
cd ~/Documents/langchain-learning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Run `source .venv/bin/activate` again every time you open a new terminal
tab to work on this.)

### 3. Environment file

```bash
cp .env.example .env
```

Open `.env` and paste your Groq key into `GROQ_API_KEY`. Leave
`LANGSMITH_TRACING=false` until you get to step 05.

### 4. (For step 05) Create a free LangSmith account

1. Go to https://smith.langchain.com and sign up (no card needed).
2. Settings → API Keys → Create API Key.
3. Paste it into `.env` as `LANGSMITH_API_KEY`, and set
   `LANGSMITH_TRACING=true`.

Check current free-tier trace limits at https://smith.langchain.com/pricing
since these do shift over time — but the free plan is comfortably enough
for a learning project like this one.

## Running each step

```bash
python 01_llm_basics/01_hello_llm.py
python 02_prompts_chains/02_prompt_chain.py
python 03_tools_agent/03_tool_agent.py
python 04_langgraph_agent/04_langgraph_agent.py
python 05_langsmith_tracing/05_tracing_demo.py     # after LangSmith setup
python 06_rag_agent/06_rag_agent.py                # first run downloads a small embedding model (~90MB)

python 07_persistence/07_persistence.py
python 08_human_in_the_loop/08_human_in_the_loop.py   # will pause and wait for you to type y/n
python 09_streaming/09_streaming.py
python 10_multi_agent/10_multi_agent.py
python 11_structured_state/11_structured_state.py
python 12_time_travel/12_time_travel.py
python 13_langsmith_eval/13_langsmith_eval.py      # requires a real LANGSMITH_API_KEY in .env
```

Read each script top to bottom before running it — the docstring at the
top explains the one new concept it introduces, and the code is commented
at every non-obvious line. Don't just run them; open them in an editor.

## Using Jupyter instead of the scripts

Everything above also works as a notebook — see `notebook/walkthrough.ipynb`,
which mirrors all six steps as cells you can run and rerun individually.
This is often the nicer way to learn: you can inspect an `AIMessage` object,
tweak a prompt, and re-run just that cell instead of the whole script.

```bash
# from the project root, with the venv activated (jupyterlab is in requirements.txt)
jupyter lab
```

This opens Jupyter in your browser. Navigate to `notebook/walkthrough.ipynb`
and open it. Make sure it's using this project's virtual environment as its
kernel (Kernel menu -> Change Kernel), otherwise it'll use your system
Python and fail on missing packages — the notebook's first cell has the
one-time command to register the venv as a Jupyter kernel if it's not
showing up as an option.

## What to do after finishing all 13

Ideas to extend this into something portfolio-worthy (you already have a
GitHub portfolio effort in progress — this fits naturally alongside it):

1. **Swap the fake `get_weather` / `get_usage_metrics` tools for real data**
   (Open-Meteo needs no key for weather; for the metrics agent in step 10,
   point it at something real you have access to) — teaches you real tool
   error handling.
2. **Combine steps 07 and 08**: add human-in-the-loop approval to a graph
   that also has persistence, so an approval request survives even if you
   close and reopen the process.
3. **Add a third specialist to step 10's supervisor** — e.g. an agent that
   drafts a Slack-style message from the writer agent's summary — to see
   how the routing logic scales past two.
4. **Try `llama-3.3-70b-versatile`** for a quality comparison against
   `openai/gpt-oss-20b`, especially on tool-calling reliability in steps
   03/04/10.
5. **Write up steps 08, 10, and 13 as a short blog post / GitHub README**
   with screenshots from LangSmith (traces and the eval experiment table) —
   concrete, demonstrable evidence of hands-on agent-framework experience
   for a resume or interview, and more distinctive than a generic tutorial
   project since few candidates show evaluation and human-in-the-loop work.
6. **Adapt step 10's supervisor pattern into something tied to your actual
   work** — a small internal Q&A agent over AICT/adoption metrics, with a
   metrics-lookup agent and a summarizer agent, is a natural next project
   once you're comfortable with everything here.

## Troubleshooting

- **`AuthenticationError` / 401** — check `GROQ_API_KEY` in `.env` is set
  and you ran `source .venv/bin/activate` in this terminal tab.
- **`429 Too Many Requests`** — you hit Groq's free-tier rate or monthly
  limit. Wait a bit, or switch `GROQ_MODEL` to `openai/gpt-oss-20b` if
  you were on the rate-limited 70b model.
- **Model ignores tools / never calls them** — Groq's Llama models
  generally support tool calling well; if it's inconsistent, try lowering
  `temperature` further (already 0 in steps 3–4) or rephrasing the prompt
  to be more explicit about when to use a tool.
- **`ModuleNotFoundError`** — you probably forgot to
  `source .venv/bin/activate` in this terminal tab.
- **Step 06 is slow the first time** — it's downloading the embedding
  model from Hugging Face once; subsequent runs use the local cache.
