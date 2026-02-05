"""
Main graph construction for the Calendar Agent.
"""

from langgraph.graph import END, START, StateGraph
from .utils import (calendar_agent_node, tool_nodes, route, CalendarAgentState)


def create_calendar_graph():
    """创建并编译 Calendar Agent 图。

    图结构 (ReAct 模式):
    ```
    START
      │
      ▼
    calendar_agent ◄────────┐
      │                     │
      ├─► tools ────────────┘
      │
      └─► __end__
    ```

    Returns:
        编译后的图
    """
    # 创建状态图
    workflow = StateGraph(CalendarAgentState)

    # 添加节点
    workflow.add_node("calendar_agent", calendar_agent_node)
    workflow.add_node("tools", tool_nodes)

    # 添加边
    workflow.add_conditional_edges(
        "calendar_agent",
        route,
        {"tools": "tools", END: END}
    )
    workflow.add_edge(START, "calendar_agent")
    workflow.add_edge("tools", "calendar_agent")

    return workflow.compile()
