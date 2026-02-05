"""
Tools for the Paper Analysis Agent.

提供论文分析功能，包括结构分析、关键信息提取、摘要生成等。
"""

from langchain.tools import tool
from typing import Optional, List, Dict
import json
import re
import os
from pathlib import Path


# 论文存储目录（与搜索代理的下载目录保持一致）
PAPER_STORAGE_PATH = os.getenv("ARXIV_STORAGE_PATH", "E:\\PYTHON\\scholar-agent\\data\\arxiv")


@tool
def check_paper_exists(paper_id: str) -> str:
    """检查指定论文是否已下载到本地。

    在分析论文之前，必须先调用此工具检查论文是否存在。
    如果论文不存在，分析代理应该返回提示信息，让主代理调用搜索代理下载论文。

    Args:
        paper_id: 论文ID（arXiv ID，如 "2401.12345" 或论文标题的一部分）

    Returns:
        JSON格式的检查结果，包含：
        - exists: 是否存在
        - paper_path: 论文路径（如果存在）
        - message: 提示信息
    """
    storage_path = Path(PAPER_STORAGE_PATH)
    
    result = {
        "exists": False,
        "paper_id": paper_id,
        "paper_path": None,
        "available_papers": [],
        "message": ""
    }
    
    # 检查存储目录是否存在
    if not storage_path.exists():
        result["message"] = f"论文存储目录不存在: {PAPER_STORAGE_PATH}。请先下载论文。"
        result["action_required"] = "DOWNLOAD_PAPER"
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    # 搜索匹配的论文文件
    paper_files = []
    for ext in ["*.md", "*.pdf", "*.txt"]:
        paper_files.extend(storage_path.glob(f"**/{ext}"))
    
    # 检查是否有匹配的论文
    matching_papers = []
    for paper_file in paper_files:
        file_name = paper_file.stem.lower()
        if paper_id.lower() in file_name or file_name in paper_id.lower():
            matching_papers.append(str(paper_file))
    
    if matching_papers:
        result["exists"] = True
        result["paper_path"] = matching_papers[0]
        result["matching_papers"] = matching_papers
        result["message"] = f"找到论文文件: {matching_papers[0]}"
        result["action_required"] = "PROCEED_ANALYSIS"
    else:
        # 列出可用的论文（最多10个）
        available = [str(f.name) for f in paper_files[:10]]
        result["available_papers"] = available
        result["message"] = f"未找到与 '{paper_id}' 匹配的论文。可用论文: {len(paper_files)} 篇。请先使用搜索代理下载该论文。"
        result["action_required"] = "DOWNLOAD_PAPER"
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def list_downloaded_papers() -> str:
    """列出所有已下载的论文。

    用于查看本地已有的论文文件列表。

    Returns:
        JSON格式的已下载论文列表
    """
    storage_path = Path(PAPER_STORAGE_PATH)
    
    result = {
        "storage_path": PAPER_STORAGE_PATH,
        "papers": [],
        "total_count": 0
    }
    
    if not storage_path.exists():
        result["message"] = "论文存储目录不存在"
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    # 搜索论文文件
    paper_files = []
    for ext in ["*.md", "*.pdf", "*.txt"]:
        paper_files.extend(storage_path.glob(f"**/{ext}"))
    
    papers_info = []
    for paper_file in paper_files[:50]:  # 限制最多50篇
        papers_info.append({
            "name": paper_file.name,
            "path": str(paper_file),
            "size_kb": round(paper_file.stat().st_size / 1024, 2)
        })
    
    result["papers"] = papers_info
    result["total_count"] = len(paper_files)
    result["message"] = f"共找到 {len(paper_files)} 篇已下载的论文"
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def read_downloaded_paper(paper_path: str) -> str:
    """读取已下载论文的内容。

    Args:
        paper_path: 论文文件的完整路径

    Returns:
        论文内容（文本格式）
    """
    try:
        paper_file = Path(paper_path)
        if not paper_file.exists():
            return json.dumps({
                "error": True,
                "message": f"论文文件不存在: {paper_path}",
                "action_required": "DOWNLOAD_PAPER"
            }, ensure_ascii=False)
        
        with open(paper_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        return content
    except Exception as e:
        return json.dumps({
            "error": True,
            "message": f"读取论文失败: {str(e)}"
        }, ensure_ascii=False)


@tool
def extract_paper_structure(paper_content: str) -> str:
    """从论文内容中提取结构信息。

    分析论文的章节结构，识别主要部分（Abstract、Introduction、Methods、Results、Discussion、Conclusion等）。

    Args:
        paper_content: 论文的文本内容（markdown 或纯文本格式）

    Returns:
        JSON格式的论文结构信息
    """
    # 常见的论文章节标题模式
    section_patterns = [
        (r"(?i)^#+\s*(abstract|摘要)", "Abstract"),
        (r"(?i)^#+\s*(introduction|引言|前言)", "Introduction"),
        (r"(?i)^#+\s*(related\s*work|background|相关工作|背景)", "Related Work"),
        (r"(?i)^#+\s*(method|methodology|approach|方法)", "Methods"),
        (r"(?i)^#+\s*(experiment|result|实验|结果)", "Results"),
        (r"(?i)^#+\s*(discussion|讨论)", "Discussion"),
        (r"(?i)^#+\s*(conclusion|结论|总结)", "Conclusion"),
        (r"(?i)^#+\s*(reference|参考文献)", "References"),
        (r"(?i)^#+\s*(appendix|附录)", "Appendix"),
    ]
    
    lines = paper_content.split('\n')
    sections = []
    current_line = 0
    
    for i, line in enumerate(lines):
        for pattern, section_name in section_patterns:
            if re.match(pattern, line.strip()):
                sections.append({
                    "section_name": section_name,
                    "original_title": line.strip(),
                    "line_number": i + 1
                })
                break
    
    # 估算论文长度
    word_count = len(paper_content.split())
    char_count = len(paper_content)
    
    result = {
        "sections": sections,
        "total_sections": len(sections),
        "word_count": word_count,
        "character_count": char_count,
        "has_abstract": any(s["section_name"] == "Abstract" for s in sections),
        "has_conclusion": any(s["section_name"] == "Conclusion" for s in sections),
    }
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def extract_key_information(paper_content: str) -> str:
    """从论文中提取关键信息。

    提取论文的核心要素，包括研究问题、方法、主要发现和贡献。

    Args:
        paper_content: 论文的文本内容

    Returns:
        JSON格式的关键信息提取结果
    """
    content_lower = paper_content.lower()
    
    # 识别研究类型
    research_types = []
    if any(kw in content_lower for kw in ["experiment", "实验", "dataset", "数据集"]):
        research_types.append("Experimental")
    if any(kw in content_lower for kw in ["survey", "综述", "review"]):
        research_types.append("Survey/Review")
    if any(kw in content_lower for kw in ["theoretical", "理论", "proof", "证明"]):
        research_types.append("Theoretical")
    if any(kw in content_lower for kw in ["case study", "案例"]):
        research_types.append("Case Study")
    
    # 识别研究领域关键词
    ai_keywords = ["machine learning", "deep learning", "neural network", "transformer", 
                   "llm", "large language model", "gpt", "bert", "attention",
                   "机器学习", "深度学习", "神经网络", "大模型", "人工智能"]
    
    bio_keywords = ["gene", "protein", "cell", "dna", "rna", "biological",
                    "基因", "蛋白质", "细胞", "生物"]
    
    found_ai_keywords = [kw for kw in ai_keywords if kw in content_lower]
    found_bio_keywords = [kw for kw in bio_keywords if kw in content_lower]
    
    # 提取可能的贡献点（基于常见表达模式）
    contribution_patterns = [
        r"(?i)(our\s+contribution|we\s+propose|we\s+introduce|our\s+main|本文提出|本文贡献|我们提出)",
        r"(?i)(novel|new\s+approach|first\s+to|创新|首次|新方法)",
    ]
    
    has_contributions = any(re.search(p, paper_content) for p in contribution_patterns)
    
    result = {
        "research_types": research_types if research_types else ["Not Identified"],
        "ai_ml_keywords_found": found_ai_keywords[:10],  # 限制数量
        "bio_keywords_found": found_bio_keywords[:10],
        "likely_has_contributions_section": has_contributions,
        "estimated_complexity": "High" if len(paper_content) > 30000 else "Medium" if len(paper_content) > 10000 else "Low",
        "note": "这是基于关键词匹配的初步分析，详细分析需要结合上下文理解"
    }
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def generate_reading_notes_template(paper_title: str, sections: Optional[str] = None) -> str:
    """生成论文阅读笔记模板。

    根据论文标题和结构信息生成结构化的阅读笔记模板。

    Args:
        paper_title: 论文标题
        sections: 可选，论文的章节结构（JSON格式）

    Returns:
        Markdown格式的阅读笔记模板
    """
    template = f"""# 论文阅读笔记

## 📄 基本信息
- **论文标题**: {paper_title}
- **阅读日期**: [填写日期]
- **作者**: [填写作者]
- **发表年份/会议/期刊**: [填写]

---

## 📝 一句话总结
[用一句话概括这篇论文的核心贡献]

---

## 🎯 研究问题
- **要解决的问题是什么？**
  [填写]

- **为什么这个问题重要？**
  [填写]

---

## 💡 核心方法
- **提出了什么方法/模型/框架？**
  [填写]

- **关键创新点是什么？**
  1. [创新点1]
  2. [创新点2]
  3. [创新点3]

---

## 📊 实验与结果
- **使用的数据集**:
  [填写]

- **主要实验结果**:
  [填写]

- **与其他方法的比较**:
  [填写]

---

## 🔍 优缺点分析

### 优点
1. [优点1]
2. [优点2]

### 局限性
1. [局限性1]
2. [局限性2]

---

## 💭 个人思考
- **可以如何应用到自己的研究中？**
  [填写]

- **有哪些后续问题值得探索？**
  [填写]

---

## 📚 相关论文
1. [相关论文1]
2. [相关论文2]

---

## 📌 重要引用
> [摘录论文中的重要语句]

"""
    return template


@tool
def compare_papers_template(paper_titles: str) -> str:
    """生成多篇论文对比分析模板。

    用于对比分析多篇相关论文。

    Args:
        paper_titles: 要对比的论文标题列表（用逗号分隔）

    Returns:
        Markdown格式的论文对比模板
    """
    titles = [t.strip() for t in paper_titles.split(",")]
    
    # 生成表格头
    header = "| 对比维度 |"
    separator = "|----------|"
    for i, title in enumerate(titles, 1):
        short_title = title[:20] + "..." if len(title) > 20 else title
        header += f" 论文{i}: {short_title} |"
        separator += "----------------------|"
    
    template = f"""# 论文对比分析

## 对比论文列表
"""
    for i, title in enumerate(titles, 1):
        template += f"{i}. {title}\n"
    
    template += f"""
---

## 📊 对比分析表

{header}
{separator}
| 研究问题 |{"  |" * len(titles)}
| 核心方法 |{"  |" * len(titles)}
| 数据集 |{"  |" * len(titles)}
| 主要结果 |{"  |" * len(titles)}
| 创新点 |{"  |" * len(titles)}
| 局限性 |{"  |" * len(titles)}
| 发表时间 |{"  |" * len(titles)}

---

## 🔍 详细对比

### 研究问题对比
[分析各论文解决的研究问题的异同]

### 方法对比
[分析各论文使用方法的异同]

### 实验设计对比
[分析实验设计的差异]

### 结果对比
[分析实验结果的差异和启示]

---

## 💡 综合分析

### 研究趋势
[从这些论文中看到的研究趋势]

### 共同发现
[多篇论文的共同发现]

### 争议点
[论文之间存在的争议或不同观点]

---

## 📝 总结
[对比分析的总体结论和启示]

"""
    return template


@tool
def analyze_abstract(abstract_text: str) -> str:
    """分析论文摘要，提取结构化信息。

    将摘要分解为背景、目的、方法、结果、结论等部分。

    Args:
        abstract_text: 论文摘要文本

    Returns:
        JSON格式的摘要分析结果
    """
    # 句子分割
    sentences = re.split(r'(?<=[.!?。！？])\s+', abstract_text.strip())
    
    # 简单的句子分类（基于位置和关键词）
    analysis = {
        "total_sentences": len(sentences),
        "word_count": len(abstract_text.split()),
        "structure_analysis": [],
        "keywords_detected": []
    }
    
    # 检测关键词
    method_keywords = ["propose", "present", "introduce", "develop", "提出", "设计", "开发"]
    result_keywords = ["achieve", "show", "demonstrate", "result", "实现", "证明", "表明", "结果"]
    background_keywords = ["recent", "existing", "traditional", "现有", "传统", "近年来"]
    
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        sent_type = "Unknown"
        
        if i == 0 or any(kw in sent_lower for kw in background_keywords):
            sent_type = "Background/Context"
        elif any(kw in sent_lower for kw in method_keywords):
            sent_type = "Method/Approach"
        elif any(kw in sent_lower for kw in result_keywords):
            sent_type = "Results/Findings"
        elif i == len(sentences) - 1:
            sent_type = "Conclusion"
        
        analysis["structure_analysis"].append({
            "sentence_index": i + 1,
            "type": sent_type,
            "preview": sent[:50] + "..." if len(sent) > 50 else sent
        })
    
    return json.dumps(analysis, ensure_ascii=False, indent=2)


# Agent 可用的工具列表
analysis_agent_tools = [
    # 论文存在性检查工具（优先调用）
    check_paper_exists,
    list_downloaded_papers,
    read_downloaded_paper,
    # 论文分析工具
    extract_paper_structure,
    extract_key_information,
    generate_reading_notes_template,
    compare_papers_template,
    analyze_abstract
]
