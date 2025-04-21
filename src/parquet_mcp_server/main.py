from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from typing import Any, Optional
from dotenv import load_dotenv
import pyarrow.parquet as pq
import mcp.types as types
import mcp.server.stdio
import pandas as pd
import numpy as np 
import requests
import asyncio
import json
import os
import logging

from parquet_mcp_server.src.search_helper import perform_search_and_scrape, find_similar_chunks

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize server
server = Server("parquet-tools")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available parquet manipulation tools."""
    return [
        types.Tool(
            name="search-web",
            description="Perform a web search and scrape results",
            inputSchema={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of search queries"
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "Page number for the search results"
                    }
                },
                "required": ["queries"]
            }
        ),
        types.Tool(
            name="extract-info-from-search",
            description="Extract relative information from previous searches",
            inputSchema={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of search queries to merge"
                    },
                },
                "required": ["queries"]
            }
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle parquet tool execution requests."""
    if not arguments:
        raise ValueError("Missing arguments")
    
    if name == "search-web":
        queries = arguments.get("queries")
        page_number = arguments.get("page_number", 1)  # Default to page 1 if not provided
        
        if not queries:
            raise ValueError("Missing queries argument")

        success, message = await perform_search_and_scrape(queries, page_number)
        return [types.TextContent(type="text", text=message)]

    elif name == "extract-info-from-search":
        queries = arguments.get("queries")

        if not queries:
            raise ValueError("Missing queries argument")

        logging.info(f"Starting extract-info-from-search with queries: {queries}")
        success, message = await find_similar_chunks(queries)
        logging.info(f"Extract-info-from-search completed with success: {success}")
        return [types.TextContent(type="text", text=message)]

    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the server using stdin/stdout streams."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="search-tools",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
