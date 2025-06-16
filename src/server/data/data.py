from model.black_forest import convert_image_to_bytes
from server.client import Client
from lib import logger
from beartype import beartype
import asyncio
import uuid
import json


@beartype
class Data:
    """Manage client queued input messages"""

    def __init__(self, client: Client):
        self.client = client

    async def manage_message(self, message):
        """Process message according to his type"""
        match message.type:
            case "chat":
                await self.message_chat(message)
            case "json":
                await self.message_json(message)
            case "image":
                await self.message_json(message)
            case "speech":
                await self.message_json(message)
            case "error":
                await self.message_json(message)
            case _:
                print("Unknown message type")


    async def message_chat(self, message):
        """Manage chat message"""
        try:
            output_generator = self.client.agent.achat(message.text, str(self.client.uid))
            async for token in output_generator:
                logger.info(f"Received token for client {self.client.get_uid()}: {token}")
                await self.client.send_message("chat", token)

            logger.info(f"Stream completed for client {self.client.get_uid()}")

        # Manage exceptions
        except asyncio.CancelledError:
            logger.info(f"Stream cancelled for client {self.client.get_uid()} for websocket {self.client.websocket.remote_address}")
            raise
        except Exception as e:
            logger.error(f"Error during chat stream: {e}")
            await self.client.send_error(500, f"Error during chat stream in thread {self.client.uid}: {e}")

    async def message_image(self, message):
        pass

    async def message_json(self, message):
        pass
