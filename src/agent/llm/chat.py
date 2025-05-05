from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger


def chat(agent, user_input: str, thread_id: int = 0):
    """Send a prompt to the LLM and receive a structured response."""
    agent_input = {"messages": [HumanMessage(content=user_input)]}
    config = {"configurable": {"thread_id": thread_id}}
    final_response_content = ""

    try:
        for token in agent.stream(
            agent_input, config=config, stream_mode="values"
        ):
            last_message = token["messages"][-1]

            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                new_content = last_message.content[len(final_response_content) :]
                if new_content:
                    print(new_content, end="", flush=True)
                    final_response_content += new_content
    except Exception as e:
        logger.info(f"\nAgent error occurred: {e}")
        return f"[Error during agent execution: {e}]"

    return final_response_content

def run(agent):
    print("Type 'exit' to quit")
    current_thread_id = 0

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                print("bye")
                break
            if not user_input:
                continue

            print("Agent: ")

            chat(agent, user_input, thread_id=current_thread_id)

        except KeyboardInterrupt:
            print("\nSession interrupted")
            break
