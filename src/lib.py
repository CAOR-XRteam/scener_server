import logging
from ollama import chat
from beartype import beartype
import json


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class Error(Exception):
    pass


@beartype
def chat_call(model_name, messages: list, logger) -> str:
    try:
        return chat(model_name, messages).message.content
    except Exception as e:
        logger.error(f"Chat API call failed: {str(e)}")
        raise Error(f"Chat API call failed: {str(e)}")


@beartype
def deserialize_from_str(s: str, logger) -> dict:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid JSON: {str(e)}\nRaw response: {s}")
        raise Error(f"Invalid JSON: {str(e)}\nRaw response: {s}")

    return res
