import asyncio
import uuid
import json

from beartype import beartype
from model.black_forest import convert_image_to_bytes
from lib import logger
from server.client import Client
from server.io.valider import (
    InputMessage,
    OutputMessage,
    OutputMessageWrapper,
)


@beartype
class Input:
    """Manage client queued input messages"""

    def __init__(self, client: Client):
        self.client = client
        self.task_loop = None

    def start(self):
        self.task_loop = asyncio.create_task(self.loop())

    async def loop(self):
        """While client keep being actif, handle input messages"""
        while self.client.is_active:
            # Handle client message
            try:
                proto = await self.client.queue.input.get() # Take the older message of the queue
                await self.handle_proto(proto)
                self.client.queue.input.task_done()
            # Manage exceptions
            except asyncio.CancelledError:
                logger.info(
                    f"Session {self.client.uid} cancelled for websocket {self.client.websocket.remote_address}"
                )
                break
            except Exception as e:
                # Optional: log or handle processing errors
                logger.error(f"Session error: {e}")
                await self.client.send_message(
                    OutputMessage(
                        status="error",
                        code=500,
                        message=f"Internal server error in thread {self.client.uid}",
                    )
                )
                break

    async def handle_proto(self, proto):
        """handle one client input message - send it to async chat"""
        logger.info(f"Received message in thread {self.client.uid}: {message}")

        # Handle received messages
        try:
            output_generator = self.client.agent.achat(message, str(self.client.uid))
            async for token in output_generator:
                logger.info(f"Received token in thread {self.client.uid}: {token}")
                responses_to_send = self._parse_agent_token(token)
                for response in responses_to_send:
                    await self.client.send_message(response)

            logger.info(f"Stream completed for thread {self.client.uid}")
        # Manage exceptions
        except asyncio.CancelledError:
            logger.info(
                f"Stream cancelled for thread {self.client.uid} for websocket {self.client.websocket.remote_address}"
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
                        message=f"Error during chat stream in thread {self.client.uid}: {e}",
                    ),
                    additional_data=None,
                )
            )

    def _parse_agent_token(self, token: str) -> list[OutputMessageWrapper]:
        """Parse an agent token and return a (list of) OutputMessageWrapper ready to be sent to the client"""

        def parse_agent_response(m: str):
            try:
                return m.split("\nFinal Answer:")
            except Exception as e:
                raise ValueError(f"Error parsing agent response: {e}")

        responses_to_send = []
        try:
            thinking, final_answer = parse_agent_response(
                token
            )  # When using stream with qwen, it returns the thinking part and the final answer as one token

            responses_to_send.append(self._thinking_response(thinking))

            json_response = json.loads(final_answer)

            match json_response.get("action"):
                case "agent_response":
                    responses_to_send.append(
                        self._agent_response(json_response.get("message"))
                    )
                case "image_generation":
                    responses_to_send.append(
                        self._image_generation_response(json_response)
                    )
                case _:
                    responses_to_send.append(
                        self._unknown_action_response(json_response.get("action"))
                    )
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
            generated_images_data = json_response.get("generated_images_data")
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

    async def loop_input(self):
        """Handle incoming messages for this specific client."""

        while self.is_active:
            # Manage messages
            try:
                async for message in self.websocket:
                    # If it's pure texte
                    if isinstance(message, str):
                        logger.info(f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} sent text data: {message}.")
                        message = InputMessage(command="chat", message=message)
                        await self.queue.input.put(message)

                        """
                        try:
                            input_message_meta = InputMessageMeta.model_validate_json(
                                message
                            )
                        except ValidationError as e:
                            logger.error(
                                f"Validation error for client {self.websocket.remote_address}: {e}"
                            )
                            await self.send_message(
                                OutputMessageWrapper(
                                    output_message=OutputMessage(
                                        status="error",
                                        code=400,
                                        action="agent_response",
                                        message=f"Invalid input: {e}",
                                    ),
                                    additional_data=None,
                                )
                            )
                            """

                    # If it's pure binary
                    elif isinstance(message, bytes):
                        pass
                        """
                        if awaitingAudio:
                            logger.info(
                                f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} sent audio data."
                            )

                            os.makedirs("media/temp_audio", exist_ok=True)
                            temp_audio_filename = (
                                f"media/temp_audio/temp_audio_{uuid.uuid4().hex}.wav"
                            )

                            with open(temp_audio_filename, "wb") as f:
                                f.write(message)

                            text = speech_to_text(temp_audio_filename)

                            logger.info(
                                f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} sent audio data converted to text: {text}"
                            )

                            await self.send_message(
                                OutputMessageWrapper(
                                    output_message=OutputMessage(
                                        status="stream",
                                        code=200,
                                        action="converted_speech",
                                        message=text,
                                    ),
                                    additional_data=None,
                                )
                            )

                            message = InputMessage(command="chat", message=text)

                            await self.queue.input.put(message)

                            continue
                        else:
                            logger.warning(
                                f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} sent binary data without awaiting audio."
                            )
                            await self.send_message(
                                OutputMessageWrapper(
                                    output_message=OutputMessage(
                                        status="error",
                                        code=400,
                                        action="agent_response",
                                        message="Unexpected binary data received.",
                                    ),
                                    additional_data=None,
                                )
                            )
                            """
                    else:
                        logger.warning(
                            f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} sent an unsupported message type: {message}"
                        )

                        await self.send_message(
                            OutputMessageWrapper(
                                output_message=OutputMessage(
                                    status="error",
                                    code=400,
                                    action="agent_response",
                                    message="Unsupported message type received.",
                                ),
                            )²
                        )
            # Manage exceptions
            except asyncio.CancelledError:
                logger.error(
                    f"Task cancelled for {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}"
                )
                break
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(
                    f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} disconnected. Reason: {e}"
                )
                break
            except Exception as e:
                logger.error(
                    f"Error with client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}: {e}"
                )
                break
            finally:
                await self.close()
