from langchain.tools import tool
import datetime
from ..sub_agent import calendar_sub_agent, search_sub_agent, analysis_sub_agent


@tool
async def call_search_agent(task_description: str) -> str:
    """调用搜索子代理执行学术论文搜索任务。
    
    该工具会根据用户的需求，自动提取论文搜索所需的所有参数，包括但不限于：
    - 搜索关键词（keywords）：用户感兴趣的主题、技术、方法等
    - 作者信息（authors）：特定作者的论文
    - 时间范围（date_range）：论文发表的时间段
    - 论文类别（categories）：如 cs.AI、cs.CL、physics 等 arXiv 类别
    - 排序方式（sort_by）：按相关性、日期等排序
    - 结果数量（max_results）：期望返回的论文数量
    
    Args:
        task_description: 任务描述，详细说明用户的搜索需求和意图。
                          例如：
                          - "搜索最近一个月关于大语言模型的论文"
                          - "查找 Yann LeCun 关于深度学习的最新研究"
                          - "搜索 2024 年发表的关于 Transformer 架构优化的论文"
                          - "查找 cs.AI 领域关于多模态学习的前 10 篇热门论文"
    
    Returns:
        搜索代理执行任务后的结果字符串，包含搜索到的论文信息。
    """
    # 构建初始状态
    initial_state = {
        "messages": [{"role": "user", "content": task_description}]
    }

    # 调用搜索子代理
    result = await search_sub_agent.ainvoke(initial_state)

    # 提取最后一条消息作为返回结果
    if result and "messages" in result and len(result["messages"]) > 0:
        last_message = result["messages"][-1]
        # 处理不同类型的消息格式
        if hasattr(last_message, "content"):
            return last_message.content
        elif isinstance(last_message, dict) and "content" in last_message:
            return last_message["content"]
        else:
            return str(last_message)

    return "搜索代理未返回有效结果。"


@tool
async def call_calendar_agent(task_description: str) -> str:
    """调用日历子代理执行日历相关任务。
    
    Args:
        task_description: 任务描述，说明需要日历代理完成的具体任务。
                          例如："查询今天的日程安排"、"添加一个明天上午10点的会议"，"查询当前时间"，"自然语言描述的时间转换成数字格式"等。
    
    Returns:
        日历代理执行任务后的结果字符串。
    """
    # 构建初始状态
    initial_state = {
        "messages": [{"role": "user", "content": task_description}]
    }
    
    # 调用日历子代理
    result = await calendar_sub_agent.ainvoke(initial_state)
    
    # 提取最后一条消息作为返回结果
    if result and "messages" in result and len(result["messages"]) > 0:
        last_message = result["messages"][-1]
        # 处理不同类型的消息格式
        if hasattr(last_message, "content"):
            return last_message.content
        elif isinstance(last_message, dict) and "content" in last_message:
            return last_message["content"]
        else:
            return str(last_message)
    
    return "日历代理未返回有效结果。"


@tool
async def call_analysis_agent(task_description: str) -> str:
    """调用论文分析子代理执行论文深度分析任务。
    
    该工具用于对学术论文进行深度分析，提取关键信息，生成结构化的阅读笔记。
    支持的分析功能包括：
    - 论文结构分析：识别论文的章节结构（Abstract、Introduction、Methods、Results、Conclusion等）
    - 关键信息提取：提取研究问题、方法论、主要发现、创新贡献
    - 阅读笔记生成：生成结构化的论文阅读笔记模板
    - 论文对比分析：为多篇论文生成对比分析框架
    - 摘要深度分析：将摘要分解为背景、方法、结果、结论等部分
    
    Args:
        task_description: 任务描述，详细说明需要分析的论文内容和分析需求。
                          例如：
                          - "分析这篇论文的结构和主要贡献：[论文内容]"
                          - "提取这篇论文的关键信息和创新点"
                          - "为这篇关于Transformer的论文生成阅读笔记模板"
                          - "对比分析这三篇论文：论文A、论文B、论文C"
                          - "深度分析这篇论文的摘要：[摘要内容]"
    
    Returns:
        论文分析代理执行任务后的结果字符串，包含分析结果或生成的模板。
    """
    # 构建初始状态
    initial_state = {
        "messages": [{"role": "user", "content": task_description}]
    }
    
    # 调用论文分析子代理
    result = await analysis_sub_agent.ainvoke(initial_state)
    
    # 提取最后一条消息作为返回结果
    if result and "messages" in result and len(result["messages"]) > 0:
        last_message = result["messages"][-1]
        # 处理不同类型的消息格式
        if hasattr(last_message, "content"):
            return last_message.content
        elif isinstance(last_message, dict) and "content" in last_message:
            return last_message["content"]
        else:
            return str(last_message)
    
    return "论文分析代理未返回有效结果。"


@tool
def get_current_time():
    """获取当前时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


supervisor_tools = [call_calendar_agent, call_search_agent, call_analysis_agent]