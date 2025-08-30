from langchain.agents import initialize_agent, AgentType
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from ..tools import calculator, date, browsing, library
from .llm import model_gemma3_4b

# --------------------
# Template for agentic AI
# --------------------

# Define tools
# --------------------
tools = [calculator, date, browsing, library]

# Load the model
# --------------------
memory = InMemorySaver()
model = model_gemma3_4b()


# Initialize the agent
# --------------------
def create():
    agent = create_react_agent(
        tools=tools,
        model=model,
        agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        checkpointer=memory,
        verbose=True,
    )
    return agent


# Run function
# --------------------
def run(agent):
    while True:
        print("hello")
        query = input("User: ")
        if query.lower() in {"exit", "quit"}:
            break
        response = agent.invoke({"input": query})
        print(f"Agent: {response['output']}")
