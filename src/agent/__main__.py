import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from api import AgentAPI

if __name__ == "__main__":
    agent = AgentAPI()
    agent.run()
