"""
Sub-agents for the Supervisor Agent.
"""

from .calendar_agent.calendar_agent import create_calendar_graph
from .search_agent.search_agent import create_search_graph
from .analysis_agent.analysis_agent import create_analysis_graph
calendar_sub_agent = create_calendar_graph()
search_sub_agent = create_search_graph()
analysis_sub_agent = create_analysis_graph()
