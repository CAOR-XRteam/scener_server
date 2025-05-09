"""
chat.py

LLM chat functions

Author: Artem
Created: 05-05-2025
Last Updated: 05-05-2025
"""

from agent.agent import Agent
from beartype import beartype
from colorama import Fore
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger


@beartype
def chat(agent: Agent, user_input: str, thread_id: str = 0):
    """Send a prompt to the LLM and receive a structured response."""
    agent_input = {"messages": [HumanMessage(content=user_input)]}
    config = {"configurable": {"thread_id": thread_id}}
    final_response_content = ""

    try:
        for token in agent.agent_executor.stream(
            agent_input, config=config, stream_mode="values"
        ):
            last_message = token["messages"][-1]

            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                new_content = last_message.content[len(final_response_content) :]
                if new_content:
                    print(new_content, end="", flush=True)
                    final_response_content += new_content
        print("")
    except Exception as e:
        logger.info(f"\nAgent error occurred: {e}")
        return f"[Error during agent execution: {e}]"

    return final_response_content


@beartype
async def achat(agent: Agent, user_input: str, thread_id: str = 0):
    """Send a prompt to the LLM and receive a structured response."""

    agent_input = {"messages": [HumanMessage(content=user_input)]}

    config = {"configurable": {"thread_id": thread_id}}

    final_response_content = ""

    try:

        async for token in agent.agent_executor.astream(
            agent_input, config=config, stream_mode="values"
        ):

            last_message = token["messages"][-1]

            if isinstance(last_message, AIMessage) and not last_message.tool_calls:

                new_content = last_message.content[len(final_response_content) :]

                if new_content:

                    yield new_content

                    final_response_content += new_content

    except Exception as e:

        logger.info(f"\nAgent error occurred: {e}")

        raise


@beartype
def run(agent: Agent):
    print("-------------------------")
    print(f"Type {Fore.RED}exit{Fore.RESET} to quit")
    print("-------------------------")

    current_thread_id = "0"

    while True:
        try:
            user_input = input(f"\n{Fore.YELLOW}You: {Fore.RESET}").strip()
            if user_input.lower() == "exit":
                print("bye")
                break
            if not user_input:
                continue

            print(f"{Fore.YELLOW}\nAgent: {Fore.RESET}")

            chat(agent, user_input, thread_id=current_thread_id)

        except KeyboardInterrupt:
            print("\nSession interrupted")
            break
