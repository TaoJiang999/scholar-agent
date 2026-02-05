from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
import os
from langgraph.prebuilt import ToolNode
from .tools import supervisor_tools
from .state import SupervisorState

llm = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    api_key=os.getenv("MODEL_API_KEY"),
    base_url=os.getenv("MODEL_BASE_URL"),
    temperature=os.getenv("MODEL_TEMPERATURE"),
    max_tokens=os.getenv("MODEL_MAX_TOKENS")
)

llm_with_tools = llm.bind_tools(supervisor_tools)


# 系统提示词
SUPERVISOR_SYSTEM_PROMPT = """你是一个 Supervisor Agent（主管代理），负责协调用户请求并调用合适的子 Agent 工具来完成任务。

## 可用的子 Agent 工具:

1. **paper_search_agent** - 论文搜索
   - 功能：从 arXiv、PubMed、bioRxiv、Google Scholar 等数据库搜索学术论文
   - 参数：task_description - 详细的搜索任务描述

2. **paper_analysis_agent** - 论文分析
   - 功能：分析论文内容、提取关键信息、生成摘要
   - 参数：task_description - 详细的分析任务描述

3. **code_assistant_agent** - 代码助手
   - 功能：编写代码、调试、解释代码
   - 参数：task_description - 详细的编程任务描述

## 工作流程:

1. 理解用户请求
2. 选择合适的工具并构造详细的 task_description
3. 调用工具执行任务
4. 整合结果并回复用户

## 重要规则:

- 调用工具时，task_description 要尽可能详细，包含所有必要信息
- 如果用户请求可以直接回答，无需调用工具
- 可以连续调用多个工具完成复杂任务
"""


def supervisor_node(state:SupervisorState):
    """Supervisor 节点：处理用户请求，决定是否调用工具。

        Args:
            state: 当前状态

        Returns:
            更新后的状态
    """
    messages = state["messages"]

    # 添加系统提示词
    all_messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + list(messages)

    # 调用 LLM（已绑定工具）
    response = llm_with_tools.invoke(all_messages)

    return {"messages": [response]}

def route(state:SupervisorState):
    """判断是否需要继续调用工具。

        Args:
            state: 当前状态

        Returns:
            下一步去向
    """
    messages = state["messages"]
    last_message = messages[-1]

    # 如果最后一条消息有工具调用，继续执行工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "__end__"

# 工具节点
tool_nodes = ToolNode(supervisor_tools)