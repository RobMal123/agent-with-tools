"""
graph.py — The core LangGraph agent

Architecture:
    [START] → call_model → (has tool calls?) → call_tools → call_model → ...
                                ↓ (no tool calls)
                             [END]

This is the standard ReAct (Reason + Act) loop pattern.
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from tools import TOOLS
from prompts import SYSTEM_PROMPT


def build_agent(model_name: str = "llama3.2", use_memory: bool = True):
    """
    Build and compile the LangGraph agent.

    Args:
        model_name: Ollama model to use. Must be pulled locally first (e.g. `ollama pull llama3.2`).
                    The model must support tool/function calling.
        use_memory: Whether to persist conversation history with MemorySaver.

    Returns:
        A compiled LangGraph app ready to invoke.
    """

    # 1. Set up the LLM and bind tools to it
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model=model_name, temperature=0, base_url=base_url)
    llm_with_tools = llm.bind_tools(TOOLS)

    # 2. Define the node functions
    def call_model(state: AgentState) -> AgentState:
        """
        The 'think' node. Calls the LLM with the full conversation history.
        The LLM either responds with text (done) or requests a tool call.
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """
        Routing function — decides the next node.
        If the last message has tool calls → go to 'tools'
        Otherwise → we're done, go to END
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # 3. Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))  # ToolNode handles running the tools

    # Set entry point
    graph.set_entry_point("call_model")

    # Add conditional edge: after call_model, check if we need tools
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {"tools": "tools", END: END},
    )

    # After tools run, always go back to call_model
    graph.add_edge("tools", "call_model")

    # 4. Compile with optional memory
    checkpointer = MemorySaver() if use_memory else None
    app = graph.compile(checkpointer=checkpointer)

    return app


# Create a default agent instance
agent = build_agent()
