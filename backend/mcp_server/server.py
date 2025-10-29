import asyncio
import sys
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Créer le serveur MCP
server = Server("lexmind")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    '''Liste les outils disponibles'''
    return [
        Tool(
            name="search_documents",
            description="Recherche dans les documents juridiques du cabinet",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requête de recherche"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Nombre de résultats",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_info",
            description="Obtenir des informations sur LexMind",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    '''Exécute un outil'''
    
    if name == "search_documents":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        
        return [
            TextContent(
                type="text",
                text=f"Recherche: {query}\n\nMode test - Backend LexMind sur port 8072.\nLe backend répond mais pas encore connecté à la recherche."
            )
        ]
    
    elif name == "get_info":
        return [
            TextContent(
                type="text",
                text="LexMind v0.3\nBackend opérationnel sur http://localhost:8072\nMCP Server connecté ✓"
            )
        ]
    
    raise ValueError(f"Outil inconnu: {name}")

async def main():
    '''Point d'entrée principal'''
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="lexmind",
                server_version="0.3.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
