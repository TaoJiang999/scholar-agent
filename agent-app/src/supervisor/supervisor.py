from langgraph.graph import END,START,StateGraph
from .utils import (supervisor_node,tool_nodes,route,SupervisorState)



def create_graph():
    """创建并编译 Supervisor Agent 图。

    图结构 (ReAct 模式):
    ```
    START
      │
      ▼
    supervisor ◄────────┐
      │                 │
      ├─► tools ────────┘
      │
      └─► __end__
    ```

    Returns:
        编译后的图
    """
    # 创建状态图
    workflow = StateGraph(SupervisorState)

    # 添加节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tools", tool_nodes)
    # 添加边
    workflow.add_conditional_edges("supervisor",route,{"tools": "tools", END: END})
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("tools", "supervisor")

    return workflow.compile()