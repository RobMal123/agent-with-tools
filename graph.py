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
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from tools import TOOLS, set_agent_model
from prompts import SYSTEM_PROMPT, VISION_SYSTEM_PROMPT
from memory import format_memories_for_prompt


def build_agent(
    model_name: str = "gemma4:e4b",
    use_memory: bool = True,
    vision_model: str | None = None,
):
    """
    Build and compile the LangGraph agent.

    Args:
        model_name: Ollama model to use for reasoning + tool calls. Must be pulled
                    locally first (e.g. `ollama pull llama3.2`). Must support tools.
        use_memory: Whether to persist conversation history with MemorySaver.
        vision_model: Ollama model used ONLY for image turns. Most tool-capable
                    Ollama models (llama3.2, the custom gemma4 build) cannot
                    actually process images, so image messages are routed to a
                    dedicated vision model instead. Defaults to the VISION_MODEL
                    env var, or "gemma3:4b". gemma3 has working vision in Ollama
                    but does NOT support tools — which is fine, image turns don't
                    need them.

    Returns:
        A compiled LangGraph app ready to invoke.
    """

    # 1. Keep tools.py in sync so structure_thoughts etc. use the right model
    set_agent_model(model_name)

    # 2. Set up the LLM and bind tools to it
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model=model_name, temperature=0, base_url=base_url)
    llm_with_tools = llm.bind_tools(TOOLS)

    # 2b. Dedicated vision model for image turns (no tools — gemma3 has none).
    #     Lazily instantiated: ChatOllama does not load the model until invoked,
    #     so users who never attach an image pay no cost.
    vision_model_name = vision_model or os.environ.get("VISION_MODEL", "gemma3:4b")
    vision_llm = ChatOllama(model=vision_model_name, temperature=0, base_url=base_url)

    # 3. Define the node functions
    def call_model(state: AgentState) -> AgentState:
        """
        The 'think' node. Calls the LLM with the full conversation history.
        The LLM either responds with text (done) or requests a tool call.

        Long-term memories are loaded fresh on every call so the LLM always
        sees the latest facts without needing a full rebuild.

        Vision routing: the primary tool-capable model usually cannot process
        images (llama3.2 is text-only; the custom gemma4 build reports a vision
        capability that does not actually work in Ollama). So when the latest
        human message contains an image, we route that single turn to a dedicated
        vision model (gemma3:4b by default) with a short, image-focused system
        prompt and no tools. Every text-only turn uses the primary model with the
        full system prompt + tools as normal.
        """
        # Detect images in the most recent human message
        _last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        _has_image = (
            _last_human is not None
            and isinstance(_last_human.content, list)
            and any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in _last_human.content
            )
        )

        if _has_image:
            # Route image turns to the dedicated vision model (no tools).
            vis_messages = (
                [SystemMessage(content=VISION_SYSTEM_PROMPT)] + list(state["messages"])
            )
            try:
                response = vision_llm.invoke(vis_messages)
            except Exception as e:
                from langchain_core.messages import AIMessage
                response = AIMessage(content=(
                    f"⚠️ Couldn't reach the vision model `{vision_model_name}`. "
                    f"Pull it with `ollama pull {vision_model_name}` (or set the "
                    f"VISION_MODEL env var to one you have). Error: {e}"
                ))
        else:
            # Normal mode: primary model with full system prompt + tools.
            memory_ctx = format_memories_for_prompt()
            system_content = (
                f"{SYSTEM_PROMPT}\n\n{memory_ctx}" if memory_ctx else SYSTEM_PROMPT
            )
            messages = [SystemMessage(content=system_content)] + list(state["messages"])
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


# Create a default agent instance. Override the model with AGENT_MODEL; gemma4:e4b is the
# validated default — it handles tool-calling and answers memory-recall questions from
# context, whereas llama3.2 mis-fired (deleting a memory when asked to recall it).
agent = build_agent(model_name=os.environ.get("AGENT_MODEL", "gemma4:e4b"))
