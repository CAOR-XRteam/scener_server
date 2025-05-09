import logging
import sys
import os
import json

from loguru import logger

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")


def load_config():
    """Load the configuration from the JSON file."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file '{CONFIG_PATH}' not found.")

    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("langgraph").setLevel(logging.DEBUG)
    logging.getLogger("langchain_core").setLevel(logging.DEBUG)


logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> | {message}",
)
