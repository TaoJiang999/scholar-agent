"""
Main graph construction for the Paper Analysis Agent.
"""

from langgraph.graph import END, START, StateGraph
from .utils import (analysis_agent_node, tool_nodes, route, PaperAnalysisAgentState)


def create_analysis_graph():
    """创建并编译 Paper Analysis Agent 图。

    图结构 (ReAct 模式):
    ```
    START
      │
      ▼
    analysis_agent ◄────────┐
      │                     │
      ├─► tools ────────────┘
      │
      └─► __end__
    ```

    Returns:
        编译后的图
    """
    # 创建状态图
    workflow = StateGraph(PaperAnalysisAgentState)

    # 添加节点
    workflow.add_node("analysis_agent", analysis_agent_node)
    workflow.add_node("tools", tool_nodes)

    # 添加边
    workflow.add_conditional_edges(
        "analysis_agent",
        route,
        {"tools": "tools", END: END}
    )
    workflow.add_edge(START, "analysis_agent")
    workflow.add_edge("tools", "analysis_agent")

    return workflow.compile()
