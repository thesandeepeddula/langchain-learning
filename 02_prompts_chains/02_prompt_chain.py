"""
STEP 2 — Prompt templates and chains (LCEL: the `|` pipe syntax).

Concept: A "chain" is just a pipeline. LangChain's LCEL (LangChain
Expression Language) lets you connect pieces with `|`, the same way you'd
pipe commands in a shell:

    prompt | model | output_parser

Each piece has a `.invoke()` method with the same signature, so they
compose. This is the single most important pattern in LangChain — nearly
everything else (agents, RAG, LangGraph nodes) is built from pieces like
this.

Run:
  python 02_prompt_chain.py
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
# from langchain_huggingface import ChatHuggingFace

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

llm = ChatGroq(model=MODEL, temperature=0.3)

# {role} and {topic} are placeholders filled in at call time.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {role}. Answer in 2 short sentences, no fluff."),
        ("human", "Explain {topic} to me."),
    ]
)

# StrOutputParser just extracts .content from the AIMessage so you get a plain string back.
chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    result = chain.invoke({"role": "senior ML engineer", "topic": "vector embeddings"})
    print(result)

    # Chains are reusable with different inputs — that's the whole point.
    print("\n---\n")
    result2 = chain.invoke({"role": "5-year-old", "topic": "vector embeddings"})
    print(result2)
