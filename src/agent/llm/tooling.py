from langchain.callbacks.base import BaseCallbackHandler
from loguru import logger
from colorama import Fore
from asyncio import Queue
import json
from pydantic import BaseModel
from typing import Literal

""" Custom tool tracker for functionnal tests """


class ToolOutput(BaseModel):
    stauts: Literal["success", "error"]
    tool_name: Literal[
        "generate_image", "initial_decomposer", "final_decomposer", "improver"
    ]
    message: str
    payload: dict | None


class Tool_callback(BaseCallbackHandler):
    def __init__(self, queue: Queue[ToolOutput]):
        self.used_tools = []
        self.final_answer_tools = (
            "final_decomposer",
            "image_generation",
            "generate_3d_object",
        )
        self.queue = queue

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Start when a tool is being to be used"""
        # Log starting tool
        tool_name = serialized.get("name")
        logger.info(f"Using tool {Fore.GREEN}{tool_name}{Fore.RESET}")

        # Store used tool names
        if tool_name and tool_name not in self.used_tools:
            self.used_tools.append(tool_name)

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Starts when a tool finishes, puts the result in the queue for further processing."""

        tool_name = kwargs.get("name")
        logger.info(
            f"Tool '{tool_name}' finished. Output is internal, not forwarding to client."
        )

        if tool_name not in self.final_answer_tools:
            return

        try:
            output_dict = json.loads(output)
            tool_output = ToolOutput(
                status="success",
                tool_name={tool_name},
                message=f"Tool '{tool_name}' completed successfully.",
                payload=output_dict,
            )
        except Exception as e:
            logger.error(f"Error in on_tool_end callback: {e}")
            tool_output = ToolOutput(
                stauts="error",
                tool_name={tool_name},
                message=str(e),
                payload=None,
            )
        finally:
            self.queue.put_nowait(tool_output)
