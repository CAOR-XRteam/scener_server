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
from sdk.messages import *
from model.black_forest import convert_image_to_bytes
from model.trellis import read_glb

# from model.trellis import read_glb

""" Custom tool tracker for functionnal tests """


class Tool_callback(BaseCallbackHandler):
    def __init__(self):
        self.used_tools = []
        self.final_answer_tools = (
            "generate_image",
            "generate_3d_object",
            "generate_3d_scene",
        )
        self.structured_response: (
            OutgoingConvertedSpeechMessage
            | OutgoingGenerated3DObjectsMessage
            | OutgoingGeneratedImagesMessage
            | OutgoingGenerated3DSceneMessage
            | OutgoingErrorMessage
        ) = None

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
        tool_output = eval(f"dict({output.content})")

        # TODO: case of error in generation
        try:
            match tool_name:
                case "generate_image":
                    payload = GenerateImageOutputWrapper(**tool_output)
                    self.structured_response = OutgoingGeneratedImagesMessage(
                        text=payload.generate_image_output.text,
                        assets=[
                            AppMediaAsset(
                                id=image_meta_data.id,
                                filename=image_meta_data.path,
                                data=convert_image_to_bytes(image_meta_data.path),
                            )
                            for image_meta_data in payload.generate_image_output.data
                        ],
                    )
                case "generate_3d_object":
                    payload = Generate3DObjectOutputWrapper(**tool_output)
                    self.structured_response = OutgoingGenerated3DObjectsMessage(
                        text=payload.generate_3d_object_output.text,
                        assets=[
                            AppMediaAsset(
                                id=payload.generate_3d_object_output.data.id,
                                filename=payload.generate_3d_object_output.data.path,
                                data=read_glb(
                                    payload.generate_3d_object_output.data.path
                                ),
                            )
                        ],
                    )
                case "generate_3d_scene":
                    pass
        except Exception as e:
            logger.error(f"Error in on_tool_end callback: {e}")
            self.structured_response = OutgoingErrorMessage(status=500, text=str(e))

    def on_tool_error(self, error: BaseException, **kwargs) -> None:
        tool_name = kwargs.get("name")
        logger.error(f"Tool '{tool_name}' encountered an error: {error}")
        self.structured_response = OutgoingErrorMessage(
            status=500, text=f"Tool '{tool_name}' failed: {error}"
        )
