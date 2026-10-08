"""
STEP 6 (capstone) — A RAG agent: retrieval-augmented generation, wired
into the LangGraph agent as just another tool.

Concept: RAG = don't make the model guess from what it memorized during
training — give it the actual source text at question time. The pattern:

    1. Split your documents into chunks.
    2. Turn each chunk into a vector (an "embedding") — a list of numbers
       that captures its meaning, so similar text has similar vectors.
    3. Store those vectors in a vector database (Chroma here — runs
       locally, no cost, no server to manage).
    4. At question time: embed the question, find the most similar chunks,
       and stuff them into the prompt as context.

The key insight for this step: retrieval is just ANOTHER TOOL from the
agent's point of view. Nothing about the LangGraph wiring from step 04
changes — you're only adding one more function to the `tools` list.

Chat goes through Groq (free API). Embeddings run locally via a small
open sentence-transformers model — this is a plain Python/PyTorch model,
not an app, so it has no macOS version requirement; it just needs the pip
packages in requirements.txt. The first run downloads the model
(~90MB) from Hugging Face, then it's cached and works offline.

Run:
  python 06_rag_agent.py
"""

import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
CHAT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DOCS_PATH = os.path.join(os.path.dirname(__file__), "docs", "servicenow_notes.txt")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), ".chroma_db")


def build_vectorstore() -> Chroma:
    loader = TextLoader(DOCS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(documents)

    # A small, fast, local embedding model — runs on CPU, no API key, no GPU needed.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


def search_notes(query: str) -> str:
    """Search the local notes for information relevant to the query. Use this
    before answering any question about AICT, CMDB, or the downsell/upsell model."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant notes found."
    return "\n---\n".join(d.page_content for d in docs)


tools = [search_notes]
llm = ChatGroq(model=CHAT_MODEL, temperature=0).bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()


if __name__ == "__main__":
    question = "What does AICT do, and how does it relate to the CMDB?"
    result = graph.invoke({"messages": [{"role": "user", "content": question}]})
    print("Q:", question)
    print("A:", result["messages"][-1].content)
    print(
        "\nIf LANGSMITH_TRACING=true in your .env, open smith.langchain.com to see "
        "the retrieval step and chat call as separate spans in the trace."
    )
