import asyncio
import websockets

from agent.api import AgentAPI
from beartype import beartype
from colorama import Fore
from lib import logger
from server.valider import InputMessage, OutputMessage
from pydantic import ValidationError


@beartype
class IO:
    """Manage client session and input / ouput messages"""

    # Main function
    def __init__(self, websocket: websockets.ServerConnection, agent: AgentAPI):
        self.websocket = websocket  # The WebSocket connection object
        self.session = None
        self.is_active = True  # State to track if the client is active

        self.queue_input = asyncio.Queue()  # Message queue for this client
        self.queue_output = asyncio.Queue()  # Message queue for this client
        self.disconnection = asyncio.Event()

        self.task_input = None
        self.task_output = None
        self.task_session = None

        self.agent = agent

    async def send_message(self, output_message: OutputMessage):
        """Queue a message to be sent to the client."""
        try:
            await self.queue_output.put(output_message)
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
            await self.queue_output.put(binary_data)
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
        while self.is_active:
            # Input management
            try:
                async for message in self.websocket:
                    message = InputMessage(command="chat", message=message)
                    await self.queue_input.put(message)
            # Exceptions
            except ValidationError as e:
                logger.error(
                    f"Validation error for client {self.websocket.remote_address}: {e}"
                )
                await self.send_message(
                    OutputMessage(
                        status="error",
                        code=400,
                        message=f"Invalid input: {e}",
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
            # Output management
            try:
                message: OutputMessage = (
                    await self.queue_output.get()
                )  
                await self.websocket.send(message.message) # Wait for a message to send
                logger.info(
                    f"Sent message to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}:\n {message}"
                )
            # Exceptions
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