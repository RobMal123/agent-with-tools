from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The state that flows through the graph.
    `messages` uses add_messages reducer — it appends
    instead of overwriting, giving us full conversation history.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
