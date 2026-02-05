"""
Node functions for the Calendar Agent.
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
import os
from langgraph.prebuilt import ToolNode
from .tools import calendar_agent_tools
from .state import CalendarAgentState

# 初始化 LLM（使用环境变量配置）
llm = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    api_key=os.getenv("MODEL_API_KEY"),
    base_url=os.getenv("MODEL_BASE_URL"),
    temperature=float(os.getenv("MODEL_TEMPERATURE")),
    max_tokens=int(os.getenv("MODEL_MAX_TOKENS"))
)

llm_with_tools = llm.bind_tools(calendar_agent_tools)


# 系统提示词
CALENDAR_AGENT_SYSTEM_PROMPT = """你是一个专门用于日历时间解析Agent。

## 主要功能：
将用户的自然语言时间描述转换为精确的日期范围。

## 可用工具：

1. **parse_natural_date** - 解析自然语言时间表达式
   - 支持中文：最近一周、过去三个月、去年、2024年上半年等
   - 支持英文：last week、past 3 months、last year、2024等
   - 返回 JSON 格式的 start_date 和 end_date

2. **get_current_datetime** - 获取当前日期和时间
   - 返回当前的完整时间信息

3. **calculate_date_range** - 计算日期差
   - 计算两个日期之间的天数差

## ⚠️ 关键规则：处理开放式时间表达

当用户使用"以来"、"至今"、"到现在"等开放式时间表达时，**必须先调用 get_current_datetime 获取当前时间**，然后将当前日期作为结束时间。

**示例**：
- "2026年以来" → 先获取当前时间，假设今天是 2026-02-05，则解析为 2026-01-01 至 2026-02-05
- "去年以来" → 先获取当前时间，解析为去年1月1日至今天
- "最近一周" → 先获取当前时间，解析为今天往前推7天

**错误示例**（避免）：
- "2026年以来" 解析为 2026-01-01 至 2026-12-31 ❌（这是错的，结束时间应该是今天）

## 工作流程：

1. 分析用户的时间描述
2. **如果是开放式时间表达（包含"以来"、"至今"、"到现在"、"最近"等词），必须先调用 get_current_datetime 获取当前时间**
3. 使用 parse_natural_date 工具解析时间表达式
4. 返回清晰的日期范围结果

## 输出格式：

当返回日期范围时，请使用以下格式：
- 开始日期：YYYY-MM-DD
- 结束日期：YYYY-MM-DD
- 简要说明解析结果

## 重要规则：

- 所有日期使用 ISO 8601 格式 (YYYY-MM-DD)
- **开放式时间表达的结束日期必须是当前日期，而不是年末或未来日期**
- 如果无法解析，返回默认的"最近一个月"范围
- 始终提供清晰的解析说明
"""


async def calendar_agent_node(state: CalendarAgentState):
    """Calendar Agent 节点：处理时间解析请求。

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    messages = state["messages"]

    # 添加系统提示词
    all_messages = [SystemMessage(content=CALENDAR_AGENT_SYSTEM_PROMPT)] + list(messages)

    # 调用 LLM（已绑定工具）
    response = await llm_with_tools.ainvoke(all_messages)

    return {"messages": [response]}


def route(state: CalendarAgentState):
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
tool_nodes = ToolNode(calendar_agent_tools)
