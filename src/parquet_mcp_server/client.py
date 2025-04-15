from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from langgraph.prebuilt import create_react_agent
from mcp.client.stdio import stdio_client
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import asyncio
import json
import os 
import logging


load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


# Initialize Ollama LangChain model
model = ChatOllama(
    base_url=os.getenv("OLLAMA_URL"),
    model="llama3.1:8b",
)

openai_model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",  # or your deployment
    api_version="2024-08-01-preview",  # or your api version
)



server_params = StdioServerParameters(
    command="uv",
    args=[
        "--directory", os.getenv("SERVER_DIRECTORY", "./src/parquet_mcp_server"),
        "run", os.getenv("MAIN_SCRIPT", "main.py")
    ],
    env={
        "EMBEDDING_URL": os.getenv("EMBEDDING_URL"),
        "OLLAMA_URL": os.getenv("OLLAMA_URL"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
        "SEARCHAPI_API_KEY": os.getenv("SEARCHAPI_API_KEY"),
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
    }
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)

            print("🔁 Interactive Agent Started. Type 'exit' to stop.")
            while True:
                user_input = input("\n🧑‍💻 Your query: ")
                if user_input.strip().lower() == "exit":
                    print("👋 Exiting.")
                    break
                # user_input = "قیمت ایفون ۱۶ از سرچ قبلی بکش بیرون"

                conversation = [HumanMessage(content=user_input)]

                while True:
                    response = await agent.ainvoke({"messages": conversation})
                    new_messages = response["messages"]
                    conversation.extend(new_messages)

                    print("\n--- 💬 New Messages ---")
                    tool_result = None
                    tool_function_name = None

                    for msg in new_messages:
                        role = "**User**" if isinstance(msg, HumanMessage) else "**AI**" if isinstance(msg, AIMessage) else "**Tool**"

                        if isinstance(msg.content, list):
                            for c in msg.content:
                                if role != "**Tool**":
                                    print(f"{role}: {c.get('text', '')}")
                        else:
                            if role != "**Tool**":
                                print(f"{role}: {msg.content}")

                        # Tool call detection
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                print(f"🔧 AI is calling tool: {tool_call['name']} with arguments: {tool_call['args']}")
                                tool_function_name = tool_call["name"]

                        # Tool response
                        if isinstance(msg, ToolMessage):
                            tool_result = msg
                            print(f"{role} (tool output preview): {msg.content[:20]}...")  # Preview

                    # 🧠 Use Ollama LLM directly for 'extract-info-from-search'
                    if tool_result:
                        prompt_content = f"This is the user input query: {user_input}\nand this is the extracted information from the internet. please answer the user query based on these information: \n{tool_result.content}"
                        print(prompt_content)
                        print("\n--- 🧠 Using OpenAI to extract info ---")

                        final_response = await openai_model.ainvoke([HumanMessage(content=prompt_content)])

                        print("\n--- ✅ Final Answer ---")
                        print("**AI (Azure OpenAI)**:", final_response.content)
                        break

                    break

            


async def perform_search_and_scrape_async(queries: list[str], page_number: int = 1) -> tuple[bool, str]:
    """
    Process a markdown file into chunks and save as parquet
    
    Args:
        queries (list[str]): List of search queries
        page_number (int, optional): Page number for the search results
    
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
                "search-web",
                {
                    "queries": queries,
                    "page_number": page_number
                }
            )
            return result
            

def perform_search_and_scrape(queries: list[str], page_number: int = 1) -> tuple[bool, str]:
    """
    Perform a web search and scrape results (synchronous version)
    
    Args:
        queries (list[str]): List of search queries
        page_number (int, optional): Page number for the search results
    
    Returns:
        tuple[bool, str]: Success status and message with output location
    """
    return asyncio.run(perform_search_and_scrape_async(queries, page_number))

async def find_similar_chunks_async(queries: list[str]):
    server_params = StdioServerParameters(
        command="uv",
        args=["--directory", "./src/parquet_mcp_server", "run", "main.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Call the find-similar-chunks tool
            result = await session.call_tool(
                "extract-info-from-search",
                {
                    "queries": queries,
                }
            )
            return result

def find_similar_chunks(queries: list[str]):
    """
    Find similar chunks based on a merged query and a JSON file of embeddings
    
    Args:
        queries (list[str]): List of search queries to merge
    
    Returns:
        The result of the similarity search
    """
    return asyncio.run(find_similar_chunks_async(queries))

if __name__ == "__main__":
    asyncio.run(main())

