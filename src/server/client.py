import asyncio
import websockets

from agent.api import AgentAPI
from beartype import beartype
from colorama import Fore
from lib import logger
from server.valider import InputMessage, OutputMessage, OutputMessageWrapper
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

        self.queue_input: asyncio.Queue[InputMessage] = (
            asyncio.Queue()
        )  # Message queue for this client
        self.queue_output: asyncio.Queue[OutputMessageWrapper] = (
            asyncio.Queue()
        )  # Message queue for this client
        self.disconnection = asyncio.Event()

        self.task_input = None
        self.task_output = None
        self.task_session = None

        self.agent = agent

    def start(self):
        from server.session import Session

        """Start input/output handlers."""
        self.session = Session(self)
        self.task_input = asyncio.create_task(self.loop_input())
        self.task_output = asyncio.create_task(self.loop_output())
        self.task_session = asyncio.create_task(self.session.run())

    async def send_message(self, output_message: OutputMessageWrapper):
        """Queue a message to be sent to the client."""
        try:
            await self.queue_output.put(
                output_message,
            )
        except asyncio.CancelledError:
            logger.error(
                f"Task was cancelled while sending message to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}, initial message: {output_message}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Error queuing message for {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}: {e}, initial message: {output_message}"
            )
            raise

    # Subfunction
    async def loop_input(self):
        """Handle incoming messages for this specific client."""
        while self.is_active:
            try:
                async for message in self.websocket:
                    message = InputMessage(command="chat", message=message)
                    await self.queue_input.put(message)
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
            try:
                message: OutputMessageWrapper = (
                    await self.queue_output.get()
                )  # Wait for a message to send
                output_message_json = message.output_message.model_dump_json()

                match message.output_message.action:
                    case "agent_response" | "thinking_process" | "unknown_action":
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

        self._clear_queues()

    def _clear_queues(self):
        """Clear queues without blocking."""
        while not self.queue_input.empty():
            try:
                self.queue_input.get_nowait()
                self.queue_input.task_done()
            except asyncio.QueueEmpty:
                break
        while not self.queue_output.empty():
            try:
                self.queue_output.get_nowait()
                self.queue_output.task_done()
            except asyncio.QueueEmpty:
                break
