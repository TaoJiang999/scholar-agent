"""
Main graph construction for the Search Agent.
"""

from langgraph.graph import END, START, StateGraph
from .utils import (search_agent_node, tool_nodes, route, SearchAgentState)


def create_search_graph():
    """创建并编译  Search Agent 图。

    图结构 (ReAct 模式):
    ```
    START
      │
      ▼
     search_agent ◄────────┐
      │                     │
      ├─► tools ────────────┘
      │
      └─► __end__
    ```

    Returns:
        编译后的图
    """
    # 创建状态图
    workflow = StateGraph(SearchAgentState)

    # 添加节点
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("tools", tool_nodes)

    # 添加边
    workflow.add_conditional_edges(
        "search_agent",
        route,
        {"tools": "tools", END: END}
    )
    workflow.add_edge(START, "search_agent")
    workflow.add_edge("tools", "search_agent")

    return workflow.compile()
