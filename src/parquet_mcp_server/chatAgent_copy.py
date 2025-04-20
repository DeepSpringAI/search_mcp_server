import asyncio
import os
from dotenv import load_dotenv
from typing import cast

import chainlit as cl
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

load_dotenv()

# Base model for demonstration
model = OpenAIChatCompletionClient(
    model="llama3.1:8b",
    base_url=os.getenv("BASE_OLLAMA_URL"),
    api_key="ollama",
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
    args=[
        "--directory",
        "/home/agent/workspace/search_mcp_server/src/parquet_mcp_server",
        "run",
        "main.py",
    ],
)

# User input function
async def user_input_func(prompt: str, cancellation_token: CancellationToken | None = None) -> str:
    try:
        response = await cl.AskUserMessage(content=prompt).send()
    except TimeoutError:
        return "User did not provide any input within the time limit."
    return response["output"] if response else "User did not provide any input."

# Recreate team dynamically based on user-selected mode
async def recreate_team_based_on_mode() -> RoundRobinGroupChat:
    mode = cl.user_session.get("mode", "scratch")
    tools = await mcp_server_tools(server_params)

    if mode == "scratch":
        fetcher_system = """
        You are a web search assistant. 
        Generate search queries from user prompt. ONE ENGLISH QUERY, ONE PERSIAN QUERY.
        Pass user queries to mcp_server and get results.
        Return "ALL" information to summarizer.
        """
    else:  # "previous"
        fetcher_system = """
        You are a data retriever. 
        Retrieve results only from previously stored searches.
        Do not generate new search queries.
        Return "ALL" information to summarizer.
        """

    fetcher = AssistantAgent(
        name="fetcher",
        system_message=fetcher_system,
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        tools=tools,
        reflect_on_tool_use=True,
        model_client_stream=True,
    )

    summarizer = AssistantAgent(
        name="summarizer",
        system_message="""
        1. Format the 'fetcher' information in a readable table, focusing on key details like prices and specifications.
        2. DO NOT GIVE INFO FROM YOURSELF.
        3. Return the pricing in both Iran and globally based on the fetcher results.
        4. DO NOT CONVERT PRICES TO EACH OTHER. ONLY ADD PRICES YOU ACTUALLY GET.
        5. After completing the above steps, hand off the conversation to the user.
        """,
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        model_client_stream=True,
    )

    user = UserProxyAgent(name="user", input_func=user_input_func)
    termination = TextMentionTermination("TERMINATE", sources=["user"])

    return RoundRobinGroupChat([fetcher, summarizer, user], termination_condition=termination)

# Streaming logic
async def run_agent_stream(user_message: str):
    loading_message = await cl.Message(content="Processing... Please wait.", author="system").send()

    team = await recreate_team_based_on_mode()
    cl.user_session.set("team", team)  # optionally update session

    streaming_response: cl.Message | None = None

    try:
        async for msg in team.run_stream(
            task=[TextMessage(content=user_message, source="user")],
            cancellation_token=CancellationToken(),
        ):
            if isinstance(msg, ModelClientStreamingChunkEvent):
                if streaming_response is None:
                    streaming_response = cl.Message(content="", author=msg.source)
                await streaming_response.stream_token(msg.content)
            elif streaming_response is not None:
                await streaming_response.send()
                streaming_response = None
            elif isinstance(msg, TaskResult):
                final_message = "Task terminated. "
                if msg.stop_reason:
                    final_message += msg.stop_reason
                await cl.Message(content=final_message).send()

    finally:
        # Once the task is complete, remove the "Processing..." message
        if loading_message:
            loading_message.content = "Processing complete! Task finished."
            await loading_message.update()



# Initial chat start: shows feature options
@cl.on_chat_start
async def start_chat() -> None:
    actions = [
        cl.Action(name="search_from_scratch", label="Search from Scratch", icon="🔍", payload={"option": "scratch"}),
        cl.Action(name="check_previous_results", label="Check Previous Results", icon="🕘", payload={"option": "previous"})
    ]
    await cl.Message(content="Hi👋, please choose one of the features.", actions=actions).send()
    cl.user_session.set("mode", "scratch")  # default mode

# Handle action: Search from Scratch
@cl.action_callback("search_from_scratch")
async def handle_search_from_scratch(action: cl.Action):
    cl.user_session.set("mode", "scratch")
    response = await cl.AskUserMessage(content="Please enter the product name you want to search for like this 'product X price'.").send()
    if response and response.get("output"):
        await run_agent_stream(response["output"])

# Handle action: Check Previous Results
@cl.action_callback("check_previous_results")
async def handle_check_previous_results(action: cl.Action):
    cl.user_session.set("mode", "previous")
    response = await cl.AskUserMessage(content="I will fetch results from your previous searches. Please enter the product name like this 'from previous search check product X price'.").send()
    if response and response.get("output"):
        await run_agent_stream(response["output"])

# Catch all for free-form user messages
@cl.on_message
async def chat(message: cl.Message) -> None:
    await run_agent_stream(message.content)
