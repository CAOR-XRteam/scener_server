from langchain.callbacks.base import BaseCallbackHandler
from loguru import logger
from colorama import Fore
from asyncio import Queue
import json
from server.valider import OutputMessage, OutputMessageWrapper
from typing import Any


""" Custom tool tracker for functionnal tests """


class Tool_callback(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        self.used_tools = []
        self.queue = queue

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Start when a tool is being to be used"""
        # Log starting tool
        tool_name = serialized.get("name")
        logger.info(f"Using tool {Fore.GREEN}{tool_name}{Fore.RESET}")

        # Store used tool names
        if tool_name and tool_name not in self.used_tools:
            self.used_tools.append(tool_name)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Starts when a tool finishes, puts the result in the queue."""

        tool_name = kwargs.get("name")
        logger.info(f"Tool '{tool_name}' finished. Adding output to queue.")

        try:
            output_dict = json.loads(output)
            message_to_send = OutputMessageWrapper(
                output_message=OutputMessage(
                    status="in_progress",
                    code=200,
                    action=f"tool_result_{tool_name}",
                    message=f"Tool '{tool_name}' completed successfully.",
                    payload=output_dict,
                ),
                additional_data=None,
            )
            self.queue.put_nowait(message_to_send)
        except Exception as e:
            logger.error(f"Error in on_tool_end callback: {e}")
            error_message = OutputMessageWrapper(
                output_message=OutputMessage(
                    status="error", code=500, action="tool_error", message=str(e)
                ),
                additional_data=None,
            )
            self.queue.put_nowait(error_message)
