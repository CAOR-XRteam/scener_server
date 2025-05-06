"""
client.py

Handles websocket clients.

Author: Nathan SV
Created: 05-05-2025
Last Updated: 05-05-2025
"""

import asyncio
import websockets
import json
import server.valider
from lib import logger

from beartype import beartype
from colorama import Fore


# Le client manage les output et la session managera les input


@beartype
class Client:
    """Manage client session and input / ouput messages"""

    # Main function
    def __init__(self, websocket):
        self.websocket = websocket  # The WebSocket connection object
        self.session = None
        self.is_active = True  # State to track if the client is active

        self.queue_input = asyncio.Queue()  # Message queue for this client
        self.queue_output = asyncio.Queue()  # Message queue for this client
        self.disconnection = asyncio.Event()

        self.task_input = None
        self.task_output = None
        self.task_session = None

    def start(self):
        from server.session import Session

        """Start input/output handlers."""
        self.session = Session(self)
        self.task_input = asyncio.create_task(self.loop_input())
        self.task_output = asyncio.create_task(self.loop_output())
        self.task_session = asyncio.create_task(self.session.run())

    async def send_message(self, status: str, code: int, message: str):
        """Create a JSON response and queue a message to be sent to the client."""
        response = {"status": status, "code": code, "message": message}
        await self.queue_output.put(json.dumps(response))

    # Subfunction
    async def loop_input(self):
        """Handle incoming messages for this specific client."""
        while self.is_active:
            try:
                async for message in self.websocket:
                    if await server.valider.check_message(self, message):
                        await self.queue_input.put(message)
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(
                    f"Client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET} disconnected. Reason: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Error with client {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}: {e}"
                )
            finally:
                self.is_active = False  # Mark the client as inactive when disconnected
                await self.close()

    async def loop_output(self):
        """Process the outgoing messages in the client's queue."""
        while self.is_active:
            try:
                message = await self.queue_output.get()  # Wait for a message to send
                await self.websocket.send(message)
                logger.info(
                    f"Sent message to {Fore.GREEN}{self.websocket.remote_address}{Fore.RESET}:\n {message}"
                )
            except asyncio.CancelledError:
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

        # List of tasks to cancel and await
        active_tasks = [
            task
            for task in [self.task_input, self.task_output, self.task_session]
            if task and not task.done()
        ]

        # Cancel each task and await them
        for task in active_tasks:
            task.cancel()  # Cancel the task

        await asyncio.gather(active_tasks, return_exceptions=True)

        # Close the WebSocket connection
        try:
            await self.websocket.close()
            self.disconnection.set()
        except Exception as e:
            logger.error(
                f"Error closing websocket connection for {self.websocket.remote_address}: {e}"
            )

        while not self.queue_input.empty():
            await self.queue_input.get()
            self.queue_input.task_done()
        while not self.queue_output.empty():
            await self.queue_output.get()
            self.queue_output.task_done()
