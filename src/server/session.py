import asyncio
import uuid
import json

from beartype import beartype
from model.black_forest import convert_image_to_bytes
from lib import logger
from server.client import Client
from server.valider import (
    InputMessage,
    OutputMessage,
    OutputMessageWrapper,
)


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
                    OutputMessage(
                        status="error",
                        code=500,
                        message=f"Internal server error in thread {self.thread_id}",
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

    def _image_generation_response(self, json_response: dict) -> OutputMessageWrapper:
        try:
            generated_images_data = json_response.get(
                "image_generation_status_json"
            ).get("generated_images_data")
            data = []
            for image_data in generated_images_data:
                image_path = image_data.get("path")
                if image_path:
                    data.append(convert_image_to_bytes(image_path))

            return OutputMessageWrapper(
                output_message=OutputMessage(
                    status="stream",
                    code=200,
                    action="image_generation",
                    message=json_response.get("message"),
                ),
                additional_data=data,
            )
        except Exception as e:
            logger.error(f"Error converting image to bytes: {e}")
            return OutputMessageWrapper(
                output_message=OutputMessage(
                    status="error",
                    code=500,
                    action="image_generation",
                    message=f"Error converting image {image_data.get("name")} to bytes: {e}",
                ),
                additional_data=None,
            )

    def _scene_description_response(self, json_response: dict) -> OutputMessageWrapper:
        try:
            return OutputMessageWrapper(
                output_message=OutputMessage(
                    status="stream",
                    code=200,
                    action="scene_generation",
                    message="Scene JSON has been generated.",
                ),
                additional_data=json_response.get("final_scene_data"),
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

    def _3d_object_generation(self, json_response: dict) -> OutputMessageWrapper:
        # TODO
        pass

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

    def _parse_agent_token(self, token: str) -> list[OutputMessageWrapper]:
        """Parse an agent token and return a (list of) OutputMessageWrapper ready to be sent to the client"""

        def parse_agent_response(m: str):
            try:
                return m.split("\nFinal Answer:")
            except Exception as e:
                raise ValueError(f"Error parsing agent response: {e}")

        def _parse_actionable_json(
            json_response: dict, responses_to_send: list[OutputMessageWrapper]
        ):
            match json_response.get("action"):
                case "agent_response":
                    responses_to_send.append(
                        self._agent_response(json_response.get("message"))
                    )
                case "image_generation":
                    responses_to_send.append(
                        self._image_generation_response(json_response)
                    )
                case "scene_generation":
                    self._scene_description_response(json_response)
                case "3d_object_generation":
                    self._3d_object_generation(json_response)
                case _:
                    responses_to_send.append(
                        self._unknown_action_response(json_response.get("action"))
                    )

        responses_to_send = []
        try:
            thinking, final_answer = parse_agent_response(
                token
            )  # When using stream with qwen, it returns the thinking part and the final answer as one token

            responses_to_send.append(self._thinking_response(thinking))

            try:
                json_response = json.loads(final_answer)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Final Answer is not a valid JSON: {final_answer}. Error: {e}"
                )
                responses_to_send.append(
                    OutputMessageWrapper(
                        output_message=OutputMessage(
                            status="error",
                            code=400,
                            action="agent_response",
                            message=f"Agent's final response is not a valid JSON.",
                        ),
                        additional_data=None,
                    )
                )
                return responses_to_send

            if "action" in json_response:
                _parse_actionable_json(json_response, responses_to_send)
            else:
                for value in json_response.values():
                    _parse_actionable_json(value, responses_to_send)

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing agent response: {e}")
            responses_to_send.append(
                OutputMessageWrapper(
                    output_message=OutputMessage(
                        status="error",
                        code=400,
                        action="agent_response",
                        message=f"Error parsing agent response: {e}",
                    ),
                    additional_data=None,
                )
            )
        except Exception as e:
            logger.error(f"Internal error: {e}")
            responses_to_send.append(
                OutputMessageWrapper(
                    output_message=OutputMessage(
                        status="error",
                        code=400,
                        action="agent_response",
                        message=f"Internal error: {e}",
                    ),
                    additional_data=None,
                )
            )

        return responses_to_send

    async def handle_message(self, input_message: InputMessage):
        """handle one client input message - send it to async chat"""
        message = input_message.message
        logger.info(f"Received message in thread {self.thread_id}: {message}")

        try:
            output_generator = self.client.agent.achat(message, str(self.thread_id))
            async for token in output_generator:
                logger.info(f"Received token in thread {self.thread_id}: {token}")
                responses_to_send = self._parse_agent_token(token)
                for response in responses_to_send:
                    await self.client.send_message(response)

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
