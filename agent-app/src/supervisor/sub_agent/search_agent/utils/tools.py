from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
MCP_SERVER_CONFIG = {
        "paper_search_server": {
        "command": "uv",
        "transport":"stdio",
        "args": [
            "run",
            "--directory",
            "E:\\PYTHON\\scholar-agent\\agent-app\\.venv\\Lib\\site-packages\\paper_search_mcp",
            "-m",
            "paper_search_mcp.server"
        ],
        "env": {
            "SEMANTIC_SCHOLAR_API_KEY": " "
        }
    }
}

MCP_SERVER_CONFIG = {
        "paper-search": {
        "transport":"stdio",
        "command": "docker",
        "args": [
        "run",
        "-i",
        "--rm",
        "mcp/paper-search"
        ]
    }
}

MCP_SERVER_CONFIG = {
    "arxiv-mcp-server": {
        "transport":"stdio",
        "command": "docker",
        "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "ARXIV_STORAGE_PATH",
        "-v",
        "E:\\PYTHON\\scholar-agent\\data\\arxiv:/local-directory",
        "mcp/arxiv-mcp-server"
        ],
      "env": {
        "ARXIV_STORAGE_PATH": "E:\\PYTHON\\scholar-agent\\data\\arxiv"
      }
    }
}

async def _fetch_tools_from_mcp():
    async def _inner():
        client = MultiServerMCPClient(MCP_SERVER_CONFIG)
        tools = await client.get_tools()
        return tools
    return await _inner()


search_agent_tools = []
async def get_mcp_tools():
    """获取MCP工具"""
    tools= await _fetch_tools_from_mcp()
    # print(f"获取到的工具：{tools}")
    search_agent_tools.extend(tools)

asyncio.run(get_mcp_tools())