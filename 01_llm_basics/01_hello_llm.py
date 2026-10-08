"""
STEP 1 — Talk to an LLM with LangChain.

Concept: LangChain wraps every model provider (OpenAI, Anthropic, Groq,
Ollama, ...) behind the same `BaseChatModel` interface, so `.invoke()`
always looks the same no matter which model is underneath. Here the
provider is Groq — a free, fast, cloud API. No local model to install or
run, so there are no hardware/OS requirements at all.

Before running:
  1. Create a free account:    https://console.groq.com (no card needed)
  2. API Keys -> Create API Key
  3. Copy .env.example to .env in the project root and paste your key into
     GROQ_API_KEY.

Run:
  python 01_hello_llm.py
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # reads .env if present

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# ChatGroq reads GROQ_API_KEY from the environment automatically.
llm = ChatGroq(model=MODEL, temperature=0.2)

if __name__ == "__main__":
    response = llm.invoke("In one sentence, what is LangChain used for?")
    print("\nModel:", MODEL)
    print("Response:", response.content)

    # A ChatModel's response is an AIMessage object, not a plain string.
    # It carries metadata too — useful once you start debugging with LangSmith.
    print("\nMessage type:", type(response).__name__)
    print("Metadata:", response.response_metadata)
