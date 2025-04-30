from langchain.agents import initialize_agent, AgentType
from ..tools import calculator, date, browsing
from .llm import load_llm


# Define all tools
tools = [
    calculator,
    date,
    browsing
]

# Load the Gemma model
llm = load_llm()

# Initialize the agent
def create():
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True
    )

def run(agent):
    while True:
        query = input("User: ")
        if query.lower() in {"exit", "quit"}:
            break
        response = agent.run(query)
        print(f"Agent: {response}")
