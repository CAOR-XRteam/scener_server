from agent.agent import Agent
from agent.llm.tooling import Tool_callback
from beartype import beartype
from colorama import Fore
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger


@beartype
def chat(agent: Agent, query: str, thread_id: str = 0):
    """Send a prompt to the LLM and receive a structured response."""

    callback = Tool_callback()
    agent_input = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [callback]}
    final_response_content = ""

    try:
        for token in agent.executor.stream(
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
async def achat(agent: Agent, query: str, thread_id: str = 0):
    """Send a prompt to the LLM and receive a structured response."""

    callback = Tool_callback()
    agent_input = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [callback]}
    final_response_content = ""

    try:

        async for token in agent.executor.astream(
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
def ask(agent: Agent, query: str, thread_id: str = 0):
    """Send a prompt to the LLM and receive a structured response."""
    callback = Tool_callback()
    agent_input = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [callback]}

    try:
        response = agent.executor.invoke(agent_input, config=config)

        # Extraire le dernier AIMessage
        messages = response.get("messages", [])
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        if ai_messages:
            final_content = ai_messages[-1].content
            print(final_content)
        else:
            print("No AIMessage found.")

    except Exception as e:
        logger.info(f"\nAgent error occurred: {e}")
        return f"[Error during agent execution: {e}]"

    return {"answer": final_content, "tools": callback.used_tools}


@beartype
def run(agent: Agent):
    print("-------------------------")
    print(f"Type {Fore.RED}exit{Fore.RESET} to quit")
    print("-------------------------")

    current_thread_id = "0"

    while True:
        try:
            query = input(f"\n{Fore.YELLOW}You: {Fore.RESET}").strip()
            if query.lower() == "exit":
                print("bye")
                break
            if not query:
                continue

            print(f"{Fore.YELLOW}\nAgent: {Fore.RESET}")

            chat(agent, query, thread_id=current_thread_id)

        except KeyboardInterrupt:
            print("\nSession interrupted")
            break
