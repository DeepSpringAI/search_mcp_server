import asyncio
import os
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import glob
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, cast

import chainlit as cl
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import HandoffTermination
from autogen_agentchat.messages import HandoffMessage


load_dotenv()


model = OpenAIChatCompletionClient(
    model="llama3.1:8b",
    base_url=os.getenv("BASE_OLLAMA_URL"),
    api_key="ollama",  # pragma: allowlist secret
    model_info={
        "model": "llama3.1:8b",
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
    },
)





# Define server parameters
server_params = StdioServerParams(
    command="uv",
    args=["--directory", "/home/agent/workspace/PharmaAI/web_search/search_mcp_server/src/parquet_mcp_server", "run", "main.py"],
)


async def user_input_func(prompt: str, cancellation_token: CancellationToken | None = None) -> str:
    """Get user input from the UI for the user proxy agent."""
    try:
        response = await cl.AskUserMessage(content=prompt).send()
    except TimeoutError:
        return "User did not provide any input within the time limit."
    if response:
        return response["output"]  # type: ignore
    else:
        return "User did not provide any input."




query_results = []

@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    global query_results
    # Load model configuration and create the model client.
    tools = await mcp_server_tools(server_params)

    

    # Create the fetcher agent with Azure OpenAI
    fetcher = AssistantAgent(
        name="fetcher",
        system_message="""
        You are a search assistant, Search user query using your tools BEFORE ANY RESPONSE.
        Return "ALL" information to summarizer.
        """,
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        tools=tools,
        reflect_on_tool_use=True,
        model_client_stream=True,


    )      


    # Create the summarizer agent
    summarizer = AssistantAgent(
            name="summarizer",
            system_message="""
            1. Format the 'fetcher' information in a readable table, focusing on key details like prices and specifications.
            2. DO NOT GIVE INFO FROM YOUR SELF.
            3. Return the pricing in both Iran and globally based on the fetcher results.
            5. DO NOT CONVERT PRICES TO EACH OTHER. ONLY ADD PRICES YOU ACTUALLY GET.
            4. After completing the above steps, hand off the conversation to the user.

            """,
            model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
            model_client_stream=True,

        )
    # Create the user proxy agent.
    user = UserProxyAgent(
        name="user",
        input_func=user_input_func, # Uncomment this line to use user input as text.
        # input_func=user_action_func,  # Uncomment this line to use user input as action.
    )

        # Create termination condition
    termination = TextMentionTermination("TERMINATE", sources=["user"])

        # Create the team with all three agents
    group_chat = RoundRobinGroupChat([fetcher,summarizer, user], termination_condition=termination)


    # Set the assistant agent in the user session.
    cl.user_session.set("prompt_history", "")  # type: ignore
    cl.user_session.set("team", group_chat)  # type: ignore
    cl.user_session.set("first_prompt_processed", False)



# @cl.set_starters  # type: ignore
# async def set_starts() -> List[cl.Starter]:
#     return [
#         cl.Starter(
#             label="Poem Writing",
#             message="Write a poem about the ocean.",
#         ),
#         cl.Starter(
#             label="Story Writing",
#             message="Write a story about a detective solving a mystery.",
#         ),
#         cl.Starter(
#             label="Write Code",
#             message="Write a function that merge two list of numbers into single sorted list.",
#         ),
#     ]


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    
    # Get the team from the user session.
    team = cast(RoundRobinGroupChat, cl.user_session.get("team"))  # type: ignore
    first_prompt_processed = cl.user_session.get("first_prompt_processed") is True
    prompt_response = message.content

    if not first_prompt_processed:
        query_generator_agent = AssistantAgent(
            name="query_generator",
            model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
            system_message=(
                'Generate search queries from user prompt.'
                'Use small phrasses and generate it in Persian and english language. One for each.'
                'Do not generate more than 2 query in total.'
                'Put all neccessary information about the product in the search query.'
                'Return output as a list like below:'
                "['search query 1','search query 2']"
            ),
    # retries=10,
        )
        query_results = await query_generator_agent.on_messages(
            [TextMessage(content=prompt_response, source="user")],
            cancellation_token=CancellationToken(),
        )
        print("*******************************" ,query_results)
        prompt_response = query_results.chat_message.content
        cl.user_session.set("first_prompt_processed", True)


    # Streaming response message.
    streaming_response: cl.Message | None = None
    # Stream the messages from the team.
    async for msg in team.run_stream(
        task=[TextMessage(content=prompt_response, source="user")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(msg, ModelClientStreamingChunkEvent):
            # Stream the model client response to the user.
            if streaming_response is None:
                # Start a new streaming response.
                streaming_response = cl.Message(content="", author=msg.source)
            await streaming_response.stream_token(msg.content)
        elif streaming_response is not None:
            # Done streaming the model client response.
            # We can skip the current message as it is just the complete message
            # of the streaming response.
            await streaming_response.send()
            # Reset the streaming response so we won't enter this block again
            # until the next streaming response is complete.
            streaming_response = None
        elif isinstance(msg, TaskResult):
            # Send the task termination message.
            final_message = "Task terminated. "
            if msg.stop_reason:
                final_message += msg.stop_reason
            await cl.Message(content=final_message).send()
        else:
            # Skip all other message types.
            pass





# <---------- with pydanticSearcher ------------>
# import asyncio
# import os
# from openai import AsyncAzureOpenAI

# from dotenv import load_dotenv
# from pydantic_ai import Agent
# from pydantic_ai.mcp import MCPServerStdio
# from pydantic_ai.models.openai import OpenAIModel
# from pydantic_ai.providers.openai import OpenAIProvider
# import chainlit as cl

# load_dotenv()

# # Set DISPLAY environment variable for X server
# os.environ["DISPLAY"] = ":1"
# # client = AsyncAzureOpenAI(
# #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
# #     api_version=os.getenv("AZURE_OPENAI_VERSION"),
# #     api_key=os.getenv("AZURE_OPENAI_API_KEY")
# # )



# model = OpenAIModel(
#     model_name='llama3.1:8b', provider=OpenAIProvider(base_url='http://213.163.75.244:50442/v1')
# )

# print("*****************************************************************************8")
# server = MCPServerStdio(
#     command="uv",
#     args=["--directory", "/home/agent/workspace/PharmaAI/web_search/search_mcp_server/src/parquet_mcp_server", "run", "main.py"],  
# )



# agent = Agent(
#         name="product_researcher",
#         model=model,
#         mcp_servers=[server],
#         system_prompt="""
#         You are a web search assistant.Pass user prompt to mcp_server and get results.
#         convert user prompt into comma seperated list of search queries one in persian and one in English.
#         return results as a table of the information.
#         """,
#     )


# # ‼️ DO NOT include explanations, markdown, or commentary. Only return raw JSON.

# @cl.on_chat_start  # type: ignore
# async def start_chat() -> None:
#     """
#     Initialize the chat system and loads the configuration."""
   
#     res = await cl.AskUserMessage(
#         content="Hi👋, enter the product name you want to buy."
#     ).send()

# @cl.on_message
# async def handle_message(message: cl.Message):
#     try:
#         async with agent.run_mcp_servers():
#             result = await agent.run(message.content)
#             await cl.Message(content=result.data).send()
#     except Exception as e:
#         await cl.Message(content=f"❌ Error: {str(e)}").send()

# # async def main() -> None:
# #     """Run main function."""
# #     async with agent.run_mcp_servers():
# #         result = await agent.run('iphone 16 pro price')
# #     print(result.data)


# # asyncio.run(main())


