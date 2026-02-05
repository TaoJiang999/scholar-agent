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

1. **call_calendar_agent** - 日历/时间代理
   - 功能：处理时间相关的任务，包括：
     - 将自然语言时间描述转换为具体日期范围（如"最近一周" → "2026-01-29 至 2026-02-05"）
     - 查询当前时间、日程安排
     - 时间计算和格式转换
   - 参数：task_description - 时间相关的任务描述

2. **call_search_agent** - 论文搜索代理
   - 功能：从 arXiv、Google Scholar 等数据库搜索学术论文
   - 参数：task_description - 详细的搜索任务描述，应包含：
     - 搜索关键词
     - **具体的时间范围**（必须是具体日期，如 "2026-01-29 至 2026-02-05"）
     - 作者、论文类别等其他筛选条件

## ⚠️ 关键工作流程规则:

### 当用户请求包含自然语言时间描述时（如"最近一周"、"近一个月"、"今年"等），必须按以下顺序执行：

**第一步：调用 call_calendar_agent**
- 任务：将用户的自然语言时间描述转换为具体的日期范围
- 示例：用户说"最近一周"，调用日历代理获取具体日期如 "2026-01-29 至 2026-02-05"

**第二步：调用 call_search_agent**
- 任务：使用第一步获取的具体日期范围，结合用户的其他搜索条件，构造完整的搜索任务描述
- 示例 task_description："搜索从 2026-01-29 到 2026-02-05 期间发表的关于高分辨率空间转录组(high-resolution spatial transcriptomics)和空间域识别(spatial domain identification)的论文"

### 示例调用链：

**用户请求**："请查询最近一周关于大语言模型的论文"

**正确的执行流程**：
1. 首先调用 call_calendar_agent，task_description = "将'最近一周'转换为具体的日期范围"
2. 获取结果（如：2026-01-29 至 2026-02-05）
3. 然后调用 call_search_agent，task_description = "搜索 2026-01-29 至 2026-02-05 期间发表的关于大语言模型(Large Language Model, LLM)的学术论文"

## 重要规则:

- **时间处理优先**：如果用户请求中包含任何自然语言时间描述，必须先调用日历代理进行转换
- 调用搜索代理时，task_description 必须包含具体的日期范围，而不是自然语言时间描述
- task_description 要尽可能详细，包含所有必要信息
- 如果用户请求可以直接回答，无需调用工具
- 可以连续调用多个工具完成复杂任务
"""


async def supervisor_node(state:SupervisorState):
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
    response = await llm_with_tools.ainvoke(all_messages)

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