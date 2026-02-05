"""
Tools for the Calendar Agent.

提供自然语言时间解析功能，用于学术论文搜索场景。
"""

from langchain.tools import tool
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re
from typing import Optional
import json


def _get_current_time() -> datetime:
    """获取当前时间"""
    return datetime.now()


def _parse_chinese_time_expression(text: str, reference_time: datetime) -> dict:
    """解析中文时间表达式
    
    Args:
        text: 中文时间表达式
        reference_time: 参考时间（通常为当前时间）
    
    Returns:
        包含 start_date 和 end_date 的字典
    """
    text = text.strip().lower()
    today = reference_time.date()
    
    result = {
        "start_date": None,
        "end_date": today.isoformat(),
        "parsed_expression": text
    }
    
    # 最近/过去 + 数字 + 时间单位
    patterns = [
        # 最近一周/过去一周
        (r"(最近|过去|近)(\d+|一|二|三|四|五|六|七|八|九|十)?(周|星期)", "weeks"),
        # 最近一个月/过去三个月
        (r"(最近|过去|近)(\d+|一|二|三|四|五|六|七|八|九|十)?个?月", "months"),
        # 最近一年/过去两年
        (r"(最近|过去|近)(\d+|一|二|三|四|五|六|七|八|九|十)?年", "years"),
        # 最近几天/过去几天
        (r"(最近|过去|近)(\d+|一|二|三|四|五|六|七|八|九|十)?天", "days"),
    ]
    
    chinese_nums = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        None: 1, "": 1
    }
    
    for pattern, unit in patterns:
        match = re.search(pattern, text)
        if match:
            num_str = match.group(2) if match.lastindex >= 2 else None
            if num_str and num_str.isdigit():
                num = int(num_str)
            else:
                num = chinese_nums.get(num_str, 1)
            
            if unit == "weeks":
                start = today - timedelta(weeks=num)
            elif unit == "months":
                start = today - relativedelta(months=num)
            elif unit == "years":
                start = today - relativedelta(years=num)
            elif unit == "days":
                start = today - timedelta(days=num)
            
            result["start_date"] = start.isoformat()
            return result
    
    # 去年
    if "去年" in text:
        last_year = today.year - 1
        result["start_date"] = f"{last_year}-01-01"
        result["end_date"] = f"{last_year}-12-31"
        return result
    
    # 今年
    if "今年" in text:
        result["start_date"] = f"{today.year}-01-01"
        result["end_date"] = today.isoformat()
        return result
    
    # 上个月
    if "上个月" in text or "上月" in text:
        last_month = today - relativedelta(months=1)
        first_day = last_month.replace(day=1)
        last_day = (first_day + relativedelta(months=1)) - timedelta(days=1)
        result["start_date"] = first_day.isoformat()
        result["end_date"] = last_day.isoformat()
        return result
    
    # 本月/这个月
    if "本月" in text or "这个月" in text:
        first_day = today.replace(day=1)
        result["start_date"] = first_day.isoformat()
        result["end_date"] = today.isoformat()
        return result
    
    # YYYY年（上半年/下半年）
    year_match = re.search(r"(\d{4})年(上半年|下半年)?", text)
    if year_match:
        year = int(year_match.group(1))
        half = year_match.group(2)
        if half == "上半年":
            result["start_date"] = f"{year}-01-01"
            result["end_date"] = f"{year}-06-30"
        elif half == "下半年":
            result["start_date"] = f"{year}-07-01"
            result["end_date"] = f"{year}-12-31"
        else:
            result["start_date"] = f"{year}-01-01"
            result["end_date"] = f"{year}-12-31"
        return result
    
    return result


def _parse_english_time_expression(text: str, reference_time: datetime) -> dict:
    """解析英文时间表达式
    
    Args:
        text: 英文时间表达式
        reference_time: 参考时间
    
    Returns:
        包含 start_date 和 end_date 的字典
    """
    text = text.strip().lower()
    today = reference_time.date()
    
    result = {
        "start_date": None,
        "end_date": today.isoformat(),
        "parsed_expression": text
    }
    
    # last X days/weeks/months/years
    patterns = [
        (r"last\s+(\d+)\s+days?", "days"),
        (r"last\s+(\d+)\s+weeks?", "weeks"),
        (r"last\s+(\d+)\s+months?", "months"),
        (r"last\s+(\d+)\s+years?", "years"),
        (r"past\s+(\d+)\s+days?", "days"),
        (r"past\s+(\d+)\s+weeks?", "weeks"),
        (r"past\s+(\d+)\s+months?", "months"),
        (r"past\s+(\d+)\s+years?", "years"),
    ]
    
    for pattern, unit in patterns:
        match = re.search(pattern, text)
        if match:
            num = int(match.group(1))
            if unit == "days":
                start = today - timedelta(days=num)
            elif unit == "weeks":
                start = today - timedelta(weeks=num)
            elif unit == "months":
                start = today - relativedelta(months=num)
            elif unit == "years":
                start = today - relativedelta(years=num)
            
            result["start_date"] = start.isoformat()
            return result
    
    # Simple patterns
    if "last week" in text:
        result["start_date"] = (today - timedelta(weeks=1)).isoformat()
        return result
    
    if "last month" in text:
        result["start_date"] = (today - relativedelta(months=1)).isoformat()
        return result
    
    if "last year" in text:
        last_year = today.year - 1
        result["start_date"] = f"{last_year}-01-01"
        result["end_date"] = f"{last_year}-12-31"
        return result
    
    if "this year" in text:
        result["start_date"] = f"{today.year}-01-01"
        return result
    
    if "this month" in text:
        result["start_date"] = today.replace(day=1).isoformat()
        return result
    
    # Year pattern (e.g., "2024", "in 2024")
    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        year = int(year_match.group(1))
        result["start_date"] = f"{year}-01-01"
        result["end_date"] = f"{year}-12-31"
        return result
    
    return result


@tool
def parse_natural_date(time_expression: str) -> str:
    """将自然语言时间表达式解析为日期范围。

    专门用于学术论文搜索场景，支持中英文时间表达。

    Args:
        time_expression: 自然语言时间表达式，例如：
            - 中文: "最近一周", "过去三个月", "去年", "2024年上半年"
            - 英文: "last week", "past 3 months", "last year", "2024"

    Returns:
        JSON格式的日期范围，包含 start_date 和 end_date (ISO 8601格式)
    """
    reference_time = _get_current_time()
    
    # 尝试中文解析
    result = _parse_chinese_time_expression(time_expression, reference_time)
    
    # 如果中文解析失败，尝试英文解析
    if result["start_date"] is None:
        result = _parse_english_time_expression(time_expression, reference_time)
    
    # 如果都失败，返回默认值（最近一个月）
    if result["start_date"] is None:
        today = reference_time.date()
        result["start_date"] = (today - relativedelta(months=1)).isoformat()
        result["end_date"] = today.isoformat()
        result["note"] = "无法解析时间表达式，使用默认值（最近一个月）"
    
    result["reference_time"] = reference_time.isoformat()
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def get_current_datetime() -> str:
    """获取当前日期和时间。

    返回当前的日期和时间信息，用于时间计算参考。

    Returns:
        JSON格式的当前时间信息
    """
    now = _get_current_time()
    return json.dumps({
        "datetime": now.isoformat(),
        "date": now.date().isoformat(),
        "time": now.time().isoformat(),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "weekday": now.strftime("%A"),
        "weekday_cn": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
    }, ensure_ascii=False, indent=2)


@tool
def calculate_date_range(start_date: str, end_date: Optional[str] = None) -> str:
    """计算两个日期之间的天数差。

    Args:
        start_date: 开始日期 (ISO 8601格式, 如 "2024-01-01")
        end_date: 结束日期 (ISO 8601格式)，默认为当前日期

    Returns:
        JSON格式的日期差信息
    """
    try:
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date() if end_date else _get_current_time().date()
        
        delta = end - start
        
        return json.dumps({
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "days_difference": delta.days,
            "weeks_difference": round(delta.days / 7, 1),
            "months_difference": round(delta.days / 30, 1)
        }, ensure_ascii=False, indent=2)
    except ValueError as e:
        return json.dumps({
            "error": f"日期格式错误: {str(e)}",
            "expected_format": "YYYY-MM-DD"
        }, ensure_ascii=False, indent=2)


# Agent 可用的工具列表
calendar_agent_tools = [
    parse_natural_date,
    get_current_datetime,
    calculate_date_range
]
