"""
client.py

Handles websocket clients.

Author: Nathan SV
Created: 05-05-2025
Last Updated: 05-05-2025
"""

import asyncio
import websockets

import utils
import json
import valider
import session


class Client:
    """Manage client session and input / ouput messages"""

    # Main function
    def __init__(self, websocket):
        self.websocket = websocket  # The WebSocket connection object
        self.queue_input = asyncio.Queue()  # Message queue for this client
        self.queue_output = asyncio.Queue()  # Message queue for this client
        self.disconnection = asyncio.Event()
        self.session = core.session.Session(self)
        self.is_active = True  # State to track if the client is active
        self.task_input = None
        self.task_output = None
        self.task_session = None

    def start(self):
        """Start input/output handlers."""
        self.task_input = asyncio.create_task(self.loop_input())
        self.task_output = asyncio.create_task(self.loop_output())
        self.task_session = asyncio.create_task(self.session.run())

    async def send_message(self, status, code, message):
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
                utils.logger.error(
                    f"Client {utils.color.GREEN}{self.websocket.remote_address}{utils.color.RESET} disconnected. Reason: {e}"
                )
            except Exception as e:
                utils.logger.error(
                    f"Error with client {utils.color.GREEN}{self.websocket.remote_address}{utils.color.RESET}: {e}"
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
                utils.logger.info(
                    f"Sent message to {utils.color.GREEN}{self.websocket.remote_address}{utils.color.RESET}:\n {message}"
                )
            except asyncio.CancelledError:
                break  # Break out of the loop if the task is canceled
            except Exception as e:
                utils.logger.error(
                    f"Error sending message to {utils.color.GREEN}{self.websocket.remote_address}{utils.color.RESET}: {e}"
                )
                self.is_active = (
                    False  # If there’s an error, mark the client as inactive
                )

    async def close(self):
        """Close the WebSocket connection gracefully."""
        self.is_active = False

        # List of tasks to cancel and await
        tasks = [self.task_input, self.task_output, self.task_session]

        # Cancel each task and await them
        for task in tasks:
            if task:
                task.cancel()  # Cancel the task
                try:
                    await task  # Await cancellation
                except asyncio.CancelledError:
                    pass  # Ignore the cancellation error

        # Close the WebSocket connection
        await self.websocket.close()
        self.disconnection.set()
