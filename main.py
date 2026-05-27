"""
api/main.py — FastAPI server for the agent

Endpoints:
    POST /chat           → single-turn or multi-turn chat
    GET  /chat/stream    → streaming response (SSE)
    GET  /health         → health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import uuid
import json
import sys
import os
from dotenv import load_dotenv

load_dotenv()  # must run before graph/tools are imported so env vars are set
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import agent

app = FastAPI(
    title="AI Research Agent",
    description="A LangGraph-powered agent with web search, code execution, and file reading.",
    version="1.0.0",
)


# --- Request / Response models ---

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None  # Pass same thread_id to continue a conversation


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    tool_calls_made: list[str]  # Names of tools used, for transparency


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Single request-response chat.
    Pass the same thread_id to maintain conversation history.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extract the final text reply
    final_message = result["messages"][-1]
    reply = final_message.content

    # Collect names of any tools that were used
    tool_calls_made = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc["name"])

    return ChatResponse(
        reply=reply,
        thread_id=thread_id,
        tool_calls_made=list(set(tool_calls_made)),
    )


@app.get("/chat/stream")
def chat_stream(message: str, thread_id: str | None = None):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    The agent streams tokens as they're generated.
    
    Usage:
        GET /chat/stream?message=What+is+LangGraph&thread_id=abc123
    """
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        for event in agent.stream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]
            # Only stream the final assistant text tokens
            if hasattr(last_msg, "content") and last_msg.content:
                data = json.dumps({"token": last_msg.content, "thread_id": thread_id})
                yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
