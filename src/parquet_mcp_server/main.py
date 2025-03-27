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

from src.embedding_helper import get_embedding

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
                    }
                },
                "required": ["input_path", "output_path", "column_name", "embedding_column"]
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

        if not all([input_path, output_path, column_name, embedding_column]):
            raise ValueError("Missing required arguments")

        try:
            # Read the parquet file into a pandas dataframe
            df = pd.read_parquet(input_path)
            
            # Check if the column exists
            if column_name not in df.columns:
                return [types.TextContent(
                    type="text",
                    text=f"Error: Column '{column_name}' not found in the parquet file."
                )]

            # Apply embedding to each row in the specified column
            embeddings = []
            for text in df[column_name]:
                embedding = get_embedding(str(text), embedding_url)  # Make sure to convert to string if it's not
                embeddings.append(embedding)

            # Add the embeddings as a new column
            df[embedding_column] = embeddings

            # Save the modified dataframe to a new Parquet file
            df.to_parquet(output_path)

            return [types.TextContent(
                type="text",
                text=f"Successfully added embedding to column '{embedding_column}' and saved the output to {output_path}"
            )]

        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error processing parquet file: {str(e)}"
            )]

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
