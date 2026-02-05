"""
Node functions for the Search Agent.
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
import os
from langgraph.prebuilt import ToolNode
from .tools import search_agent_tools
from .state import SearchAgentState


# 初始化 LLM（使用环境变量配置）
llm = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    api_key=os.getenv("MODEL_API_KEY"),
    base_url=os.getenv("MODEL_BASE_URL"),
    temperature=float(os.getenv("MODEL_TEMPERATURE")),
    max_tokens=int(os.getenv("MODEL_MAX_TOKENS"))
)

# 注意：不在模块顶层绑定工具，因为此时 search_agent_tools 可能还是空列表
# 工具绑定将在 search_agent_node 函数内部动态执行

SYSTEM_PROMPT = '''你是一个专业的学术文献搜索助手（Search Agent）。你的主要职责是根据用户的搜索目的和意图，智能地调用相关工具完成学术文献搜索任务。

## 你的能力
1. **理解用户意图**：准确分析用户的搜索需求，识别关键信息
2. **参数提取**：从用户的自然语言描述中提取搜索所需的参数，包括：
   - 搜索关键词（keywords）
   - 作者名称（authors）
   - 时间范围（date_range）
   - 论文类别/领域（categories）
   - 排序方式（sort_by）
   - 结果数量限制（max_results）
3. **工具调用**：根据提取的参数，选择并调用合适的搜索工具
4. **结果整理**：对搜索结果进行整理和总结，以清晰的格式呈现给用户

## 可用的搜索工具
你可以使用以下工具进行文献搜索和阅读：
- **search_papers**：在 arXiv 上搜索论文，支持高级过滤和查询优化
- **download_paper**：下载并将 arXiv 论文转换为可读的 markdown 格式，用于分析和阅读
- **list_papers**：列出所有已下载和转换的论文，可立即阅读和分析
- **read_paper**：读取已下载论文的完整文本内容（以干净的 markdown 格式）

## ⚠️ 重要规则
1. **不要检查时间范围**：无论用户提供的时间范围是过去还是未来，都直接调用搜索工具执行搜索。时间范围的有效性由搜索工具自行处理。
2. **必须调用工具**：收到搜索请求后，必须立即调用相应的搜索工具，不要询问用户是否要修改搜索条件。
3. **直接执行搜索**：不要对用户的搜索参数进行额外的验证或质疑，直接使用这些参数调用工具。

## 工作流程
1. 分析用户的搜索请求，提取搜索关键词
2. **立即调用**搜索工具（如 search_arxiv、search_pubmed 等）
3. 整理搜索结果，提供论文标题、作者、摘要、链接等信息
4. 如果搜索结果为空，告知用户并建议调整关键词

## 注意事项
- 始终以用户的搜索目标为导向
- 搜索时使用英文关键词以获取更多结果
- 使用中文与用户交流
'''

async def search_agent_node(state: SearchAgentState):
    """Search Agent 节点：处理搜索请求。

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    messages = state["messages"]

    # 添加系统提示词
    all_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    # print(f"search_agent_tools: {search_agent_tools}")
    # 动态绑定工具（确保此时 MCP 工具已加载完成）
    llm_with_tools = llm.bind_tools(search_agent_tools)
    
    # 调用 LLM（已绑定工具）
    response = await llm_with_tools.ainvoke(all_messages)

    return {"messages": [response]}

def route(state: SearchAgentState):
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
tool_nodes = ToolNode(search_agent_tools)