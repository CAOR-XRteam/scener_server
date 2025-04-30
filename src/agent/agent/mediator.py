from langchain.agents import initialize_agent, AgentType

from ..tools import calculator, date, browsing, library
from .llm import model_qwen3_4b

#--------------------
#Template for agentic AI
#--------------------

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
agent = model_qwen3_4b(tools)

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
