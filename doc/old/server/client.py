import asyncio
import os
import uuid
import websockets

from agent.api import AgentAPI
from beartype import beartype
from colorama import Fore
from lib import logger, speech_to_text
from server.io.valider import (
    InputMessage,
    InputMessageMeta,
    OutputMessage,
    OutputMessageWrapper,
)
from pydantic import ValidationError


# Le client manage les output et la session managera les input


@beartype
class Client:
    """Manage client session and input / ouput messages"""

    # Main function
    def __init__(self, websocket: websockets.ServerConnection, agent: AgentAPI):
        self.websocket = websocket  # The WebSocket connection object
        self.session = None
        self.is_active = True  # State to track if the client is active
        self.disconnection = asyncio.Event()

        self.task_input = None
        self.task_output = None
        self.task_session = None

        self.agent = agent

    def start(self):
        """Start input/output handlers."""
        from server.session import Session
        
        self.session = Session(self)
        self.task_input = asyncio.create_task(self.loop_input())
        self.task_output = asyncio.create_task(self.loop_output())
        self.task_session = asyncio.create_task(self.session.run())

    async def send_message(self, output_message: OutputMessageWrapper):
        """Queue a message to be sent to the client."""
        # Queue message
        try:
            await self.queue.output.put(output_message,)
        # Manage exceptions
        except asyncio.CancelledError:
            logger.error(
                f"Task was cancelled while sending message to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}, initial message: {output_message}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Error queuing message for {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}: {e}, initial message: {output_message}"
            )

    async def send_blob(self, path: str):
        try:
            with open(path, "rb") as f:
                binary_data = f.read()
            await self.queue.output.put(binary_data)
        except asyncio.CancelledError:
            logger.error(
                f"Task was cancelled while sending message to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}, initial message: {path}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Error queuing message for {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}: {e}, initial message: {path}"
            )

    # Subfunction
    async def loop_input(self):
        """Handle incoming messages for this specific client."""

        # possible timeouts in case when one of the markers is True but the next message is never received should probably be handled by websockets library
        awaitingAudio = False
        awaitingText = False

        while self.is_active:
            assert not (
                awaitingAudio and awaitingText
            ), "State violation: awaitingAudio and awaitingText cannot both be true"
            try:
                async for message in self.websocket:
                    if isinstance(message, str):
                        if awaitingText:
                            assert (
                                not awaitingAudio
                            ), "State violation: awaitingAudio and awaitingText cannot both be true"
                            logger.info(
                                f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} sent text data: {message}."
                            )

                            awaitingText = False

                            message = InputMessage(command="chat", message=message)

                            await self.queue.input.put(message)

                            continue

                        try:
                            input_message_meta = InputMessageMeta.model_validate_json(
                                message
                            )
                            match input_message_meta.type:
                                case "audio":
                                    logger.info(
                                        f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} is sending audio data."
                                    )

                                    awaitingAudio = True
                                    awaitingText = False

                                    continue
                                case "text":
                                    logger.info(
                                        f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} is sending text data."
                                    )

                                    awaitingAudio = False
                                    awaitingText = True

                                    continue
                        except ValidationError as e:
                            logger.error(
                                f"Validation error for client {self.websocket.remote_address}: {e}"
                            )

                            awaitingText = False
                            awaitingAudio = False

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
                    elif isinstance(message, bytes):
                        if awaitingAudio:
                            assert (
                                not awaitingText
                            ), "State violation: awaitingAudio and awaitingText cannot both be true"
                            logger.info(
                                f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} sent audio data."
                            )

                            awaitingAudio = False

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
                    else:
                        logger.warning(
                            f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} sent an unsupported message type: {message}"
                        )

                        awaitingText = False
                        awaitingAudio = False

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

    async def loop_output(self):
        """Process the outgoing messages in the client's queue."""
        while self.is_active:
            # Output management
            try:
                message: OutputMessageWrapper = (
                    await self.queue.output.get()
                )  # Wait for a message to send
                output_message_json = message.output_message.model_dump_json()

                match message.output_message.action:
                    case (
                        "agent_response"
                        | "thinking_process"
                        | "converted_speech"
                        | "unknown_action"
                    ):
                        await self.websocket.send(output_message_json)
                        logger.info(
                            f"Sent message to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}:\n {output_message_json}"
                        )
                    case "image_generation":
                        if message.additional_data:
                            try:
                                await self.websocket.send(output_message_json)
                                logger.info(
                                    f"Sent message to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}:\n {output_message_json}"
                                )
                                await self.websocket.send(
                                    str(len(message.additional_data))
                                )
                                logger.info(
                                    f"Sent message to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}:\n {str(len(message.additional_data))}"
                                )
                                for image in message.additional_data:
                                    await self.websocket.send(image)
                                logger.info(
                                    f"Sent binary data to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error sending image to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}: {e}"
                                )
                        else:
                            await self.websocket.send(output_message_json)
                            logger.info(
                                f"Sent message to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}:\n {output_message_json}"
                            )
            except asyncio.CancelledError:
                logger.info(
                    f"Task cancelled for {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}"
                )
                break  # Break out of the loop if the task is canceled
            except Exception as e:
                logger.error(
                    f"Error sending message to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}: {e}"
                )
                self.is_active = (
                    False  # If there’s an error, mark the client as inactive
                )

    async def close(self):
        """Close the WebSocket connection gracefully."""
        if not self.is_active:
            return

        logger.info(f"Closing connection for {self.websocket.remote_address}")

        self.is_active = False

        tasks_to_cancel = [
            t
            for t in [self.task_input, self.task_output, self.task_session]
            if t and not t.done()
        ]
        for task in tasks_to_cancel:
            task.cancel()

        try:
            await self.websocket.close()
        except Exception as e:
            logger.error(
                f"Error closing websocket connection for {self.websocket.remote_address}: {e}"
            )
        finally:
            self.disconnection.set()

        self.queue.clear()
