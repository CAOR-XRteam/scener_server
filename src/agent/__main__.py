import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from agent.api import AgentAPI
from lib import setup_logging


if __name__ == "__main__":
    setup_logging()
    # TODO: check if the model is available and pull it otherwise
    model_name = (
        input("Enter the model name: ").strip() or "qwen3:8b"
    )  # for the momemnt managed to get it work only with qwen3:8b
    agent = AgentAPI(model_name)
    agent.run()
