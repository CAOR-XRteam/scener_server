from langchain.agents import initialize_agent, AgentType
from loguru import logger
from .tools import calculator, date, search_engine, list_assets, update_asset, image_analysis
from .llm import model, format


# Define tools
#--------------------
tools = [
    calculator,
    date,
    search_engine,
    list_assets,
    update_asset,
    image_analysis
]

# Configure agent
#--------------------
agent = model.qwen3_4b(tools)
role = "You are a helpful assistant. Please provide concise and accurate responses. do not use emojis"

# Run function
#--------------------
def run():
    config = {"configurable": {"thread_id": "1"}}
    logger.success(f"Start chatbot...")
    while True:
        # User query
        query = input("User: ")
        if query.lower() in {"exit", "quit"}:
            break

        # Agent resposne
        message = {"messages": [{"role": "system", "content": role}, {"role": "user", "content": query}]}
        response = agent.invoke(message, config)
        print(f"Agent: {response["messages"][-1].pretty_print()}")
        #print_answer(response)
