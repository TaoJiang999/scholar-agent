from dotenv import load_dotenv
load_dotenv(r"E:\PYTHON\scholar-agent\agent-app\.env")
from supervisor.supervisor import create_graph
agent = create_graph()

if __name__ == '__main__':
    # print(os.getenv("MODEL_TEMPERATURE"))
    async def main():
        response = await agent.ainvoke({"messages": [{"role":"user","content":"请你查询一下2025年以来的关于大模型，强化学习，对齐技术的最新论文，需要5篇。"}]})
        for m in response["messages"]:
            m.pretty_print()
    import asyncio
    asyncio.run(main()) 