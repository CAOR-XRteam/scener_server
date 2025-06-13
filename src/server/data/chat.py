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
class Chat:
    """Manage client queued input messages"""

    def __init__(self, client: Client):
        self.client = client

    async def handle_chat(self, message):
        """handle one client input message - send it to async chat"""
        pass
