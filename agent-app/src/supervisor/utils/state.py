"""
State definition for the Supervisor Agent.
"""

from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SupervisorState(TypedDict):
    """Supervisor Agent 的状态定义。

    Attributes:
        messages: 对话消息历史
    """

    # 消息历史
    messages: Annotated[Sequence[BaseMessage], add_messages]
