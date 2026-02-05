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
   - 功能：从 arXiv 搜索学术论文，并使用 download_paper 工具下载论文
   - 参数：task_description - 详细的搜索/下载任务描述
   - **下载论文时**：使用 download_paper 工具（需提供 paper_id 如 "2401.12345"）
   - **可能的返回状态**：
     - 下载成功：返回论文内容
     - 正在下载/转换中：返回包含 "downloading"、"converting"、"processing" 等状态信息
     - 下载失败：返回错误信息

3. **call_analysis_agent** - 论文分析代理
   - 功能：对学术论文进行深度分析
   - 参数：task_description - 论文分析任务描述，应包含论文标识（ID或标题）
   - **注意**：分析代理会检查论文是否已下载
     - 如果论文不存在，返回 `[PAPER_NOT_FOUND]` 标记
     - 如果论文存在，返回分析结果

## ⚠️ 关键工作流程规则:

### 当用户请求包含自然语言时间描述时：

**第一步：调用 call_calendar_agent**
- 任务：将自然语言时间描述转换为具体的日期范围

**第二步：调用 call_search_agent**
- 任务：使用具体日期范围进行搜索

### 当用户需要分析论文时：

**第一步：调用 call_analysis_agent**
- 任务：根据用户需求分析论文内容
- 示例 task_description："分析论文 2401.12345 的结构和主要贡献"

**第二步：检查分析代理的返回结果**

| 返回结果 | 下一步操作 |
|---------|----------|
| 正常分析结果 | 任务完成，向用户展示结果 |
| `[PAPER_NOT_FOUND]` | 调用搜索代理下载论文 |

**第三步：如果论文不存在，调用 call_search_agent 下载**
- 示例 task_description："下载论文 2401.12345"

**第四步：检查搜索代理的下载返回结果**

| 返回结果 | 下一步操作 |
|---------|----------|
| 下载成功（返回论文内容） | 调用分析代理进行分析 |
| 正在下载/转换中 | **停止流程**，告知用户稍后重试 |
| 下载失败 | **停止流程**，告知用户下载失败原因 |

## ⚠️ 避免死循环的关键规则

**当搜索代理返回以下状态时，绝对不能再次调用分析代理：**
- "downloading"、"正在下载"
- "converting"、"正在转换"
- "processing"、"处理中"
- "pending"、"等待中"
- 任何表示异步处理进行中的状态

**正确处理方式**：
1. 如果返回"正在下载/转换中"，直接向用户返回提示：
   "论文正在下载和转换中，请稍后几分钟再尝试分析。您可以使用命令'分析论文 {paper_id}'来重试。"
2. **不要**再次调用分析代理，否则会造成死循环

### 完整工作流程示例：

**用户请求**："请帮我分析论文 2401.12345"

**执行流程**：
```
1. 调用 call_analysis_agent("分析论文 2401.12345")
   │
   ├─→ 返回分析结果 → 任务完成
   │
   └─→ 返回 [PAPER_NOT_FOUND] → 继续步骤2
   
2. 调用 call_search_agent("下载论文 2401.12345")
   │
   ├─→ 返回论文内容（下载成功） → 继续步骤3
   │
   ├─→ 返回"正在下载/转换中" → 停止！告知用户稍后重试
   │
   └─→ 返回下载失败 → 停止！告知用户失败原因
   
3. 调用 call_analysis_agent("分析论文 2401.12345") → 返回分析结果
```

## 重要规则:

- **时间处理优先**：自然语言时间描述必须先调用日历代理转换
- **避免死循环**：如果搜索代理返回"正在下载/转换中"，**绝对不要**再次调用分析代理
- **论文不存在时的处理**：如果分析代理返回 [PAPER_NOT_FOUND]，调用搜索代理下载
- **下载状态检查**：下载后必须检查返回状态，只有下载成功才能继续分析
- task_description 要尽可能详细，包含所有必要信息
- 如果用户请求可以直接回答，无需调用工具
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