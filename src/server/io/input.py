from server.client import Client
from server.data.data import Data
from server.protobuf import message_pb2
from lib import logger
from beartype import beartype
import asyncio


@beartype
class Input:
    """Manage client queued input messages"""

    def __init__(self, client: Client):
        self.client = client
        self.data = Data(client)
        self.task_loop = None

    def start(self):
        self.task_loop = asyncio.create_task(self.loop())

    async def loop(self):
        """While client keep being actif, handle input messages"""
        while self.client.is_active:
            # Handle client message
            try:
                message = await self.client.queue.input.get() # Take the older message of the queue
                await self.handle_message(message)
                self.client.queue.input.task_done()

            # Manage exceptions
            except asyncio.CancelledError:
                logger.info(f"Client {self.client.get_uid()} cancelled for websocket {self.client.websocket.remote_address}")
                break
            except Exception as e:
                logger.error(f"Client input error: {e}")
                await self.client.send_error(500, f"Internal server error in thread {self.client.get_uid()}")
                break

    async def handle_message(self, message):
        """handle one client input message - send it to async chat"""
        await self.data.manage_message(message)
        logger.info(f"Received message for client {self.client.get_uid()}: {message.type}")
