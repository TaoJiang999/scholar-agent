from dotenv import load_dotenv
load_dotenv(r"E:\PYTHON\scholar-agent\agent-app\.env")
from supervisor.supervisor import create_graph
agent = create_graph()
import os
if __name__ == '__main__':
    # print(os.getenv("MODEL_TEMPERATURE"))
    response = agent.invoke({"messages": [{"role":"user","content":"请你将最近一周这个时间段转为数字时间的时间段。"}]})
    for m in response["messages"]:
        m.pretty_print()