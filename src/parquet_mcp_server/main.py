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

from src.embedding_helper import get_embedding, process_parquet_file
from src.parquet_helper import get_parquet_info
from src.duckdb_helper import convert_parquet_to_duckdb
from src.postgres_helper import convert_parquet_to_postgres
from parquet_mcp_server.src.markdown_helper import process_markdown_file
from parquet_mcp_server.src.search_helper import perform_search_and_scrape

# Set up logging
logging.basicConfig(
    filename='embedding_server.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize server
server = Server("parquet-tools")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available parquet manipulation tools."""
    return [
        types.Tool(
            name="embed-parquet",
            description="Embed text in a specific column of a parquet file",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the input Parquet file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to the output Parquet file"
                    },
                    "column_name": {
                        "type": "string",
                        "description": "The name of the column containing the text to embed"
                    },
                    "embedding_column": {
                        "type": "string",
                        "description": "Name of the new column where embeddings will be saved"
                    },
                    "batch_size": {
                        "type": "int",
                        "description": "Batch size for the embedding"
                    }
                },
                "required": ["input_path", "output_path", "column_name", "embedding_column", "batch_size"]
            }
        ),
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
            name="parquet-information",
            description="Get information about a parquet file including column names, number of rows, and file size",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the Parquet file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        types.Tool(
            name="convert-to-duckdb",
            description="Convert a Parquet file to a DuckDB database",
            inputSchema={
                "type": "object",
                "properties": {
                    "parquet_path": {
                        "type": "string",
                        "description": "Path to the input Parquet file"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save the DuckDB database (optional)"
                    }
                },
                "required": ["parquet_path"]
            }
        ),
        types.Tool(
            name="convert-to-postgres",
            description="Convert a Parquet file to a PostgreSQL table with pgvector support",
            inputSchema={
                "type": "object",
                "properties": {
                    "parquet_path": {
                        "type": "string",
                        "description": "Path to the input Parquet file"
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Name of the PostgreSQL table to create or append to"
                    }
                },
                "required": ["parquet_path", "table_name"]
            }
        ),
        types.Tool(
            name="process-markdown",
            description="Convert a markdown file into structured JSON with text chunks and save as parquet",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the markdown file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the output parquet file"
                    }
                },
                "required": ["file_path", "output_path"]
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
    
    if name == "embed-parquet":
        input_path = arguments.get("input_path")
        output_path = arguments.get("output_path")
        column_name = arguments.get("column_name")
        embedding_column = arguments.get("embedding_column")
        batch_size = arguments.get("batch_size")

        if not all([input_path, output_path, column_name, embedding_column, batch_size]):
            raise ValueError("Missing required arguments")

        success, message = process_parquet_file(input_path, output_path, column_name, embedding_column, batch_size)
        return [types.TextContent(type="text", text=message)]

    elif name == "search-web":
        queries = arguments.get("queries")
        page_number = arguments.get("page_number", 1)  # Default to page 1 if not provided
        
        if not queries:
            raise ValueError("Missing queries argument")

        success, message = perform_search_and_scrape(queries, page_number)
        return [types.TextContent(type="text", text=message)]

    elif name == "parquet-information":
        file_path = arguments.get("file_path")
        
        if not file_path:
            raise ValueError("Missing file_path argument")

        success, message = get_parquet_info(file_path)
        return [types.TextContent(type="text", text=message)]

    elif name == "convert-to-duckdb":
        parquet_path = arguments.get("parquet_path")
        output_dir = arguments.get("output_dir")
        
        if not parquet_path:
            raise ValueError("Missing parquet_path argument")

        success, message = convert_parquet_to_duckdb(parquet_path, output_dir)
        return [types.TextContent(type="text", text=message)]

    elif name == "convert-to-postgres":
        parquet_path = arguments.get("parquet_path")
        table_name = arguments.get("table_name")
        
        if not all([parquet_path, table_name]):
            raise ValueError("Missing required arguments")

        success, message = convert_parquet_to_postgres(parquet_path, table_name)
        return [types.TextContent(type="text", text=message)]

    elif name == "process-markdown":
        file_path = arguments.get("file_path")
        output_path = arguments.get("output_path")
        
        if not all([file_path, output_path]):
            raise ValueError("Missing required arguments")

        success, result = process_markdown_file(file_path, output_path)
        return [types.TextContent(type="text", text=result)]  # result is error message

    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the server using stdin/stdout streams."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="parquet-tools",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
