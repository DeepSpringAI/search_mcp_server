from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from langgraph.prebuilt import create_react_agent
from mcp.client.stdio import stdio_client
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import asyncio
import json
import os 


load_dotenv()

# Initialize a ChatOpenAI model
model = ChatOllama(
    base_url=os.getenv("OLLAMA_URL"),
    model="llama3.1:8b",
    temperature=0,
)

server_params = StdioServerParameters(
    command="uv",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["--directory", "./src/parquet_mcp_server","run","main.py"],
)

# Wrap the async code inside an async function
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # print(tools)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            # agent_response = await agent.ainvoke({"messages": "please embed the column 'text' in the parquet file '/home/agent/workspace/parquet_mcp_server/input.parquet' and save the output to '/home/agent/workspace/parquet_mcp_server/src/parquet_mcp_server/output.parquet'. please use embedding as final column and also with batch size 2"})
            # agent_response = await agent.ainvoke({"messages": "Please give me some information about the parquet file '/home/agent/workspace/parquet_mcp_server/input.parquet'"})
            # agent_response = await agent.ainvoke({"messages": "Please convert the parquet file '/home/agent/workspace/parquet_mcp_server/input.parquet' to DuckDB format and save it in '/home/agent/workspace/parquet_mcp_server/db_output'"})
            # agent_response = await agent.ainvoke({"messages": "Please process the markdown file '/home/agent/workspace/parquet_mcp_server/README.md' and save the chunks to '/home/agent/workspace/parquet_mcp_server/output/readme_chunks.parquet'"})

            # print(agent_response)
            # Loop over the responses and print them
            for response in agent_response["messages"]:
                user = ""
                
                if isinstance(response, HumanMessage):
                    user = "**User**"
                elif isinstance(response, ToolMessage):
                    user = "**Tool**"
                elif isinstance(response, AIMessage):
                    user = "**AI**"
                    
                # Check if response.content is a list or just a string
                if isinstance(response.content, list):
                    for content in response.content:
                        print(f'{user}: {content.get("text", "")}')
                    continue
                print(f"{user}: {response.content}")

async def embed_parquet_async(input_path: str, output_path: str, column_name: str, embedding_column: str, batch_size: int):
    server_params = StdioServerParameters(
        command="uv",
        args=["--directory", "./src/parquet_mcp_server","run","main.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Call the embed-parquet tool
            result = await session.call_tool(
                "embed-parquet",
                {
                    "input_path": input_path,
                    "output_path": output_path,
                    "column_name": column_name,
                    "embedding_column": embedding_column,
                    "batch_size": batch_size
                }
            )
            return result

def embed_parquet(input_path: str, output_path: str, column_name: str, embedding_column: str, batch_size: int):
    """
    Embed text in a specific column of a parquet file
    
    Args:
        input_path (str): Path to the input Parquet file
        output_path (str): Path to the output Parquet file
        column_name (str): The name of the column containing the text to embed
        embedding_column (str): Name of the new column where embeddings will be saved
    
    Returns:
        The result of the embedding operation
    """
    return asyncio.run(embed_parquet_async(input_path, output_path, column_name, embedding_column, batch_size))

async def convert_to_duckdb_async(parquet_path: str, output_dir: str = None):
    server_params = StdioServerParameters(
        command="uv",
        args=["--directory", "./src/parquet_mcp_server","run","main.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Call the convert-to-duckdb tool
            result = await session.call_tool(
                "convert-to-duckdb",
                {
                    "parquet_path": parquet_path,
                    "output_dir": output_dir
                }
            )
            return result

def convert_to_duckdb(parquet_path: str, output_dir: str = None):
    """
    Convert a Parquet file to a DuckDB database
    
    Args:
        parquet_path (str): Path to the input Parquet file
        output_dir (str, optional): Directory to save the DuckDB database
    
    Returns:
        The result of the conversion operation
    """
    return asyncio.run(convert_to_duckdb_async(parquet_path, output_dir))

async def convert_to_postgres_async(parquet_path: str, table_name: str):
    server_params = StdioServerParameters(
        command="uv",
        args=["--directory", "./src/parquet_mcp_server","run","main.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Call the convert-to-postgres tool
            result = await session.call_tool(
                "convert-to-postgres",
                {
                    "parquet_path": parquet_path,
                    "table_name": table_name
                }
            )
            return result

def convert_to_postgres(parquet_path: str, table_name: str):
    """
    Convert a Parquet file to a PostgreSQL table with pgvector support
    
    Args:
        parquet_path (str): Path to the input Parquet file
        table_name (str): Name of the PostgreSQL table to create or append to
    
    Returns:
        The result of the conversion operation
    """
    return asyncio.run(convert_to_postgres_async(parquet_path, table_name))

async def chunk_markdown_async(input_path: str, output_path: str) -> tuple[bool, str]:
    """
    Process a markdown file into chunks and save as parquet
    
    Args:
        input_path (str): Path to the markdown file to process
        output_path (str): Path where to save the output parquet file
    
    Returns:
        tuple[bool, str]: Success status and message with output location
    """
    server_params = StdioServerParameters(
        command="uv",
        args=["--directory", "./src/parquet_mcp_server", "run", "main.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Call the process-markdown tool
            result = await session.call_tool(
                "process-markdown",
                {
                    "file_path": input_path,
                    "output_path": output_path
                }
            )
            return result
            

def chunk_markdown(input_path: str, output_path: str) -> tuple[bool, str]:
    """
    Process a markdown file into chunks and save as parquet (synchronous version)
    
    Args:
        input_path (str): Path to the markdown file to process
        output_path (str): Path where to save the output parquet file
    
    Returns:
        tuple[bool, str]: Success status and message with output location
    """
    return asyncio.run(chunk_markdown_async(input_path, output_path))

if __name__ == "__main__":
    asyncio.run(main())

