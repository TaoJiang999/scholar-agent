"""
Node functions for the Paper Analysis Agent.
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
import os
from langgraph.prebuilt import ToolNode
from .tools import analysis_agent_tools
from .state import PaperAnalysisAgentState


# 初始化 LLM（使用环境变量配置）
llm = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    api_key=os.getenv("MODEL_API_KEY"),
    base_url=os.getenv("MODEL_BASE_URL"),
    temperature=float(os.getenv("MODEL_TEMPERATURE")),
    max_tokens=int(os.getenv("MODEL_MAX_TOKENS"))
)


# 系统提示词
ANALYSIS_AGENT_SYSTEM_PROMPT = """你是一个专业的学术论文分析助手（Paper Analysis Agent）。你的主要职责是帮助用户深度分析学术论文，提取关键信息，生成结构化的阅读笔记。

## 你的能力

1. **论文存在性检查**：检查论文是否已下载到本地
2. **论文结构分析**：识别论文的章节结构（Abstract、Introduction、Methods、Results、Conclusion等）
3. **关键信息提取**：提取研究问题、方法论、主要发现和创新贡献
4. **阅读笔记生成**：生成结构化的论文阅读笔记模板
5. **论文对比分析**：为多篇论文生成对比分析框架
6. **摘要深度分析**：将摘要分解为背景、方法、结果、结论等部分

## 可用工具

### 论文存在性检查工具（必须优先调用）
- **check_paper_exists**：检查指定论文是否已下载到本地（必须首先调用）
- **list_downloaded_papers**：列出所有已下载的论文
- **read_downloaded_paper**：读取已下载论文的内容

### 论文分析工具
- **extract_paper_structure**：分析论文章节结构
- **extract_key_information**：提取论文关键信息（研究类型、关键词等）
- **generate_reading_notes_template**：生成论文阅读笔记模板
- **compare_papers_template**：生成多论文对比分析模板
- **analyze_abstract**：深度分析论文摘要

## ⚠️ 关键工作流程规则

### 分析论文之前，必须先检查论文是否存在！

**第一步：调用 check_paper_exists 或 list_downloaded_papers**
- 检查用户要分析的论文是否已下载到本地
- 如果返回结果中 `action_required` 为 `"DOWNLOAD_PAPER"`，说明论文不存在

**第二步：根据检查结果决定下一步**
- 如果论文**存在**（`exists: true`）：继续进行分析
- 如果论文**不存在**（`exists: false`）：**立即停止分析**，返回以下格式的消息：

```
[PAPER_NOT_FOUND]
论文未找到，需要先下载。
论文标识: {paper_id}
建议操作: 请先使用搜索代理(call_search_agent)搜索并下载该论文，然后再进行分析。
```

**重要**：如果论文不存在，你不能尝试分析，必须返回上述格式的消息，让主代理知道需要先下载论文。

## 正常工作流程（论文存在时）

1. 调用 check_paper_exists 确认论文存在
2. 调用 read_downloaded_paper 读取论文内容
3. 根据用户需求选择合适的分析工具
4. 执行分析并整理结果
5. 以清晰、结构化的方式呈现分析结果

## 输出要求

- 使用中文与用户交流
- 分析结果应当结构清晰、重点突出
- 对于复杂的论文，提供多层次的分析
- 必要时给出进一步阅读建议

## 注意事项

- **永远不要跳过论文存在性检查**
- 论文分析应当客观、准确
- 识别论文的创新点和局限性
- 帮助用户理解论文的核心贡献
- 如果论文内容不完整，说明分析的局限性
"""


async def analysis_agent_node(state: PaperAnalysisAgentState):
    """Paper Analysis Agent 节点：处理论文分析请求。

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    messages = state["messages"]

    # 添加系统提示词
    all_messages = [SystemMessage(content=ANALYSIS_AGENT_SYSTEM_PROMPT)] + list(messages)

    # 动态绑定工具
    llm_with_tools = llm.bind_tools(analysis_agent_tools)
    
    # 调用 LLM（已绑定工具）
    response = await llm_with_tools.ainvoke(all_messages)

    return {"messages": [response]}


def route(state: PaperAnalysisAgentState):
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
tool_nodes = ToolNode(analysis_agent_tools)
