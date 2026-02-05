from langchain.tools import tool
import datetime
from ..sub_agent import calendar_sub_agent


@tool
def call_calendar_agent(task_description: str) -> str:
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
    result = calendar_sub_agent.invoke(initial_state)
    
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
def get_current_time():
    """获取当前时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


supervisor_tools = [call_calendar_agent]