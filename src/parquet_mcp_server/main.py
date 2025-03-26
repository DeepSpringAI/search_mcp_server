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

# Set up logging
logging.basicConfig(
    filename='embedding_server.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize server
server = Server("parquet-tools")

# Load environment variables from .env file
load_dotenv()

# Embedding URL from the .env file
embedding_url = os.getenv("EMBEDDING_URL")
if not embedding_url:
    raise ValueError("EMBEDDING_URL not found in .env file")

def get_embedding(text: str, url) -> list:
    # Prepare the payload with a list of prompts
    texts = [text]
    model = os.getenv("EMBEDDING_MODEL") or "llama2"
    payload = {
        "model": model,
        "input": texts  # Pass all texts in a batch
    }

    logging.debug(f"Making request to {url} with payload: {payload}")
    # Send the request with SSL verification disabled
    try:
        response = requests.post(url, json=payload, verify=False)
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            logging.debug(f"Response: {result}")
            if 'data' in result and result['data'] and 'embedding' in result['data'][0]:
                embeddings = np.array(result['data'][0]['embedding'])
                return embeddings
            else:
                logging.error(f"No embeddings found in response: {result}")
                return None
        else:
            logging.error(f"Error: {response.status_code}, {response.text}")
            return None  # If there was an error, return None
    except Exception as e:
        logging.error(f"Exception during request: {str(e)}")
        return None


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
