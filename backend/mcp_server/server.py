"""LexMind MCP Server"""
from mcp import MCPServer
import asyncio

server = MCPServer("lexmind")

@server.tool(name="search_documents", description="Recherche documents")
async def search_documents(query: str, limit: int = 5):
    return {"query": query, "results": []}

def main():
    print("🧠 LexMind MCP Server")
    print("Ready for Claude Desktop")
    asyncio.run(server.run())

if __name__ == "__main__":
    main()
