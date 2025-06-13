import asyncio
import os
import uuid
import websockets

from agent.api import AgentAPI
from server.client import Client
from server.protobuf import message_pb2
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
class Output:
    """Manage client queued ouput messages"""

    # Main function
    def __init__(self, client: Client):
        self.client = client
        self.task_loop = None

    def start(self):
        self.task_loop = asyncio.create_task(self.loop())

    async def loop(self):
        """While client keep being actif, handle output messages"""
        while self.client.is_active:
            # Handle client message
            try:
                proto = await self.client.queue.output.get() # Take the older message of the queue
                await self.handle_proto(proto)
                self.client.queue.input.task_done()

            # Manage exceptions
            except asyncio.CancelledError:
                logger.info(f"Client {self.client.uid} cancelled for websocket {self.client.websocket.remote_address}")
                break
            except Exception as e:
                # Optional: log or handle processing errors
                logger.error(f"Output error: {e}")
                await self.client.send_message(
                    OutputMessage(
                        status="error",
                        code=500,
                        message=f"Internal server error in thread {self.client.uid}",
                    )
                )
                break

    async def handle_proto(self, proto):
        """Process the outgoing messages in the client's queue."""
        # Output management
        try:
            content = message_pb2.Content()
            content.ParseFromString(proto)

            print(content.type)
            print(content.json)
            print(content.status)
            print(content.error)
        except asyncio.CancelledError:
            logger.info(f"Client {self.client.uid} cancelled for websocket {self.client.websocket.remote_address}")
