from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from loguru import logger
from colorama import Fore
from asyncio import Queue
import json
from pydantic import BaseModel
from typing import Literal
from agent.tools import *
from sdk.scene import *

""" Custom tool tracker for functionnal tests """


class ToolOutput(BaseModel):
    status: Literal["success", "error"]
    tool_name: Literal[
        "generate_image", "initial_decomposer", "final_decomposer", "improver"
    ]
    message: str
    payload: FinalDecompositionOutput | GenerateImageOutput | None


class Tool_callback(BaseCallbackHandler):
    def __init__(self, queue: Queue[ToolOutput]):
        self.used_tools = []
        self.final_answer_tools = (
            "final_decomposer",
            "generate_image",
            "generate_3d_object",
        )
        self.queue = queue

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Start when a tool is being to be used"""

        # Log starting tool
        tool_name = serialized.get("name")
        logger.info(
            f"Using tool {Fore.GREEN}{tool_name}{Fore.RESET} with input {input_str}"
        )

        # Store used tool names
        if tool_name and tool_name not in self.used_tools:
            self.used_tools.append(tool_name)

    def on_tool_end(self, output: ToolMessage, **kwargs) -> None:
        """Starts when a tool finishes, puts the result in the queue for further processing."""

        tool_name = kwargs.get("name")

        logger.info(f"Tool output: {output}")

        if tool_name not in self.final_answer_tools:
            logger.info(
                f"Tool '{tool_name}' finished. Output is internal, not forwarding to client."
            )
            return

        tool_output = None

        try:
            payload = None
            match tool_name:
                case "final_decomposer":
                    payload = FinalDecompositionOutput.model_validate(
                        eval(f"dict({output.content})")
                    )
                case "generate_image":
                    payload = GenerateImageOutputWrapper.model_validate(
                        eval(f"dict({output.content})")
                    ).general_image_output

            tool_output = ToolOutput(
                status="success",
                tool_name=tool_name,
                message=f"Tool '{tool_name}' completed successfully.",
                payload=payload,
            )
        except Exception as e:
            logger.error(f"Error in on_tool_end callback: {e}")
            tool_output = ToolOutput(
                status="error",
                tool_name=tool_name,
                message=str(e),
                payload=None,
            )
        finally:
            if tool_output:
                self.queue.put_nowait(tool_output)

    def on_tool_error(self, error: BaseException, **kwargs) -> None:
        tool_name = kwargs.get("name")
        logger.error(f"Tool '{tool_name}' encountered an error: {error}")
