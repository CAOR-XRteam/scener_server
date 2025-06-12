import asyncio
import uuid
import json

from agent.tools.asset.generate_image import ImageMetaData, GenerateImageOutput
from agent.tools.asset.generate_3d_object import (
    TDObjectMetaData,
    Generate3DObjectOutput,
)
from beartype import beartype
from model.black_forest import convert_image_to_bytes
from lib import logger
from server.client import Client
from server.valider import (
    InputMessage,
    OutputMessage,
    OutputMessageWrapper,
)
from sdk.scene import *
from agent.llm.tooling import ToolOutput


@beartype
class Session:
    def __init__(self, client: Client):
        """init session by client and assign an ID"""
        self.client = client
        self.thread_id = uuid.uuid1()

    async def run(self):
        """While client keep being actif, handle input messages"""
        while self.client.is_active:
            try:
                message = await self.client.queue_input.get()
                await self.handle_message(message)
                self.client.queue_input.task_done()
            except asyncio.CancelledError:
                logger.info(
                    f"Session {self.thread_id} cancelled for websocket {self.client.websocket.remote_address}"
                )
                break
            except Exception as e:
                # Optional: log or handle processing errors
                logger.error(f"Session error: {e}")
                await self.client.send_message(
                    OutputMessageWrapper(
                        output_message=OutputMessage(
                            status="error",
                            code=500,
                            action="agent_response",
                            message=f"Internal server error in thread {self.thread_id}: {e}",
                        ),
                        additional_data=None,
                    )
                )
                break

    def _thinking_response(self, thinking: str) -> OutputMessageWrapper:
        """Create a thinking response message"""
        return OutputMessageWrapper(
            output_message=OutputMessage(
                status="stream",
                code=200,
                action="thinking_process",
                message=thinking,
            ),
            additional_data=None,
        )

    def _agent_response(self, message: str) -> OutputMessageWrapper:
        """Create an agent response message"""
        return OutputMessageWrapper(
            output_message=OutputMessage(
                status="stream",
                code=200,
                action="agent_response",
                message=message,
            ),
            additional_data=None,
        )

    def _image_generation_response(
        self, image_generation_status: GenerateImageOutput
    ) -> OutputMessageWrapper:
        data = []
        logger.info(f"Preparing to send images: {image_generation_status}")
        for image_data in image_generation_status.generated_images_data:
            try:
                image_path = image_data.path
                if image_path:
                    data.append(convert_image_to_bytes(image_path))
            except Exception as e:
                logger.error(f"Error converting image to bytes: {e}")
                return OutputMessageWrapper(
                    output_message=OutputMessage(
                        status="error",
                        code=500,
                        action="image_generation",
                        message=f"Error converting image {image_data.filename} to bytes: {e}",
                    ),
                    additional_data=None,
                )
        return OutputMessageWrapper(
            output_message=OutputMessage(
                status="stream",
                code=200,
                action="image_generation",
                message=image_generation_status.message,
            ),
            additional_data=data,
        )

    def _scene_description_response(
        self, final_decomposition_status: FinalDecompositionOutput
    ) -> OutputMessageWrapper:
        try:
            return OutputMessageWrapper(
                output_message=OutputMessage(
                    status="stream",
                    code=200,
                    action="scene_generation",
                    message="Scene JSON has been generated.",
                ),
                additional_data=final_decomposition_status.final_scene_json,
            )
        except json.JSONDecodeError as e:
            logger.error(
                f"Error extracting scene description from the agent's response: {e}"
            )
            return OutputMessageWrapper(
                output_message=OutputMessage(
                    status="error",
                    code=500,
                    action="scene_generation",
                    message=f"Error extracting scene description from the agent's response: {e}",
                ),
                additional_data=None,
            )

    def _3d_object_generation_response(
        self,
        generation_output: Generate3DObjectOutput,
    ) -> list[OutputMessageWrapper]:
        responses = []
        object_data = generation_output.generated_images_data

        responses.append(
            OutputMessageWrapper(
                output_message=OutputMessage(
                    status="stream",
                    code=200,
                    action="3d_object_generation",
                    message=generation_output.message,
                ),
                additional_data=None,
            )
        )

        for object in object_data:
            object_path = object_data.path
            if object_path:
                with open(object_path, "rb") as f:
                    glb_bytes = f.read()
            responses.extend(
                [
                    OutputMessageWrapper(
                        output_message=OutputMessage(
                            status="stream",
                            code=200,
                            action="3d_object_generation",
                            message="Generated 3D object id",
                        ),
                        additional_data=object.id,
                    ),
                    OutputMessageWrapper(
                        output_message=OutputMessage(
                            status="stream",
                            code=200,
                            action="3d_object_generation",
                            message="Gerenated 3D object bytes",
                        ),
                        additional_data=glb_bytes,
                    ),
                ]
            )
        return

    def _unknown_action_response(self, action: str) -> OutputMessageWrapper:
        """Create a response for unknown actions"""
        return OutputMessageWrapper(
            output_message=OutputMessage(
                status="error",
                code=400,
                action="unknown_action",
                message=f"Unknown action in agent response: {action}",
            ),
            additional_data=None,
        )

    def _parse_agent_response(self, m: str):
        try:
            return m.split("\nFinal Answer:")
        except Exception as e:
            raise ValueError(f"Error parsing agent response: {e}")

    async def handle_message(self, input_message: InputMessage):
        """handle one client input message - send it to async chat"""
        message = input_message.message
        logger.info(f"Received message in thread {self.thread_id}: {message}")

        tool_outputs_queue: asyncio.Queue[ToolOutput] = asyncio.Queue()

        async def handle_tool_output():
            while True:
                tool_output = await tool_outputs_queue.get()
                if tool_output is None:
                    break
                if tool_output.status == "success":
                    match tool_output.tool_name:
                        case "final_decomposer":
                            await self.client.send_message(
                                self._scene_description_response(tool_output.payload)
                            )
                        case "generate_image":
                            await self.client.send_message(
                                self._image_generation_response(tool_output.payload)
                            )
                        case "generate_3d_object":
                            responses_to_send = self._3d_object_generation_response(
                                tool_output.payload
                            )
                            for response in responses_to_send:
                                await self.client.send_message(response)
                else:
                    logger.error(f"Error during chat stream: {tool_output.message}")
                    await self.client.send_message(
                        OutputMessageWrapper(
                            OutputMessage(
                                status="error",
                                code=500,
                                action="agent_response",
                                message=f"Error during chat stream in thread {self.thread_id}: {tool_output.message}",
                            ),
                            additional_data=None,
                        )
                    )

        handle_tool_output_task = asyncio.create_task(handle_tool_output())

        try:
            output_generator = self.client.agent.achat(
                message, tool_outputs_queue, str(self.thread_id)
            )
            async for token in output_generator:
                logger.info(f"Received token in thread {self.thread_id}: {token}")
                thinking, final_answer = self._parse_agent_response(token)
                await self.client.send_message(self._thinking_response(thinking))
                await self.client.send_message(self._agent_response(final_answer))

            logger.info(f"Stream completed for thread {self.thread_id}")
        except asyncio.CancelledError:
            logger.info(
                f"Stream cancelled for thread {self.thread_id} for websocket {self.client.websocket.remote_address}"
            )
            raise
        except Exception as e:
            logger.error(f"Error during chat stream: {e}")
            await self.client.send_message(
                OutputMessageWrapper(
                    OutputMessage(
                        status="error",
                        code=500,
                        action="agent_response",
                        message=f"Error during chat stream in thread {self.thread_id}: {e}",
                    ),
                    additional_data=None,
                )
            )
        finally:
            await tool_outputs_queue.put(None)
            await handle_tool_output_task
