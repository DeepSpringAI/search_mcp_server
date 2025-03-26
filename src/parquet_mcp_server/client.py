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
    args=["run","main.py"],
)

# Wrap the async code inside an async function
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            print(tools)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "please embed the column 'text' in the parquet file 'input.parquet' and save the output to 'output.parquet'. please use embedding as final column"})

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


if __name__ == "__main__":
    asyncio.run(main())

