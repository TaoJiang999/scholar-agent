from dotenv import load_dotenv
load_dotenv(r"E:\PYTHON\scholar-agent\agent-app\.env")
from supervisor.supervisor import create_graph
agent = create_graph()

if __name__ == '__main__':
    # print(os.getenv("MODEL_TEMPERATURE"))
    async def main():
        response = await agent.ainvoke({"messages": [{"role":"user","content":"请帮我分析论文 2401.12345 的结构和主要贡献"}]})
        for m in response["messages"]:
            m.pretty_print()
    import asyncio
    asyncio.run(main()) 