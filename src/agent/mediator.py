from langchain.agents import initialize_agent, AgentType

from .tools import calculator, date, browsing, library
from .llm import model, format


# Define tools
#--------------------
tools = [
    calculator,
    date,
    browsing,
    library
]

# Load the model
#--------------------
agent = model.qwen3_4b(tools)

# Run function
#--------------------
def run():
    config = {"configurable": {"thread_id": "1"}}
    while True:
        query = input("User: ")
        if query.lower() in {"exit", "quit"}:
            break
        response = agent.invoke({"messages": [{"role": "user", "content": query}]}, config)
        print(f"Agent: {response["messages"][-1].pretty_print()}")
        #print_answer(response)
