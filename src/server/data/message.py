import os
from agent.tools.input import speech_to_text
from model.black_forest import convert_image_to_bytes
from server.client import Client
from lib import logger
from beartype import beartype
import asyncio
import uuid
import json


# Peut etre faudra til mettre chacune des data processing dans des classes distincts


@beartype
class Message:
    """Manage client queued input messages"""

    def __init__(self, client: Client):
        self.client = client

    async def manage_outcoming_message(self, message):
        """Process outcoming message according to his type"""
        match OutcomingMessageType(message.command):
            case OutcomingMessageType.UNRELATED_RESPONSE:
                await self.message_chat(message)
            case OutcomingMessageType.GENERATE_IMAGE:
                await self.message_image(message)
            case OutcomingMessageType.GENERATE_3D_OBJECT:
                await self.message_json(message)
            case OutcomingMessageType.GENERATE_3D_SCENE:
                await self.message_json(message)
            case OutcomingMessageType.CONVERT_SPEECH:
                await self.message_speech(message)
            case _:
                logger.error(
                    f"Unknown message type {message.command} for client {self.client.get_uid()}"
                )

    async def manage_incoming_message(self, message):
        """Process incoming message according to his type"""
        match IncomingMessageType(message.type):
            case IncomingMessageType.TEXT:
                await self.message_chat(message)
            case IncomingMessageType.AUDIO:
                await self.message_speech(message)
            case IncomingMessageType.GESTURE:
                await self.message_gesture(message)
            case _:
                logger.error(
                    f"Unknown message type {message.command} for client {self.client.get_uid()}"
                )

    async def message_chat(self, message):
        """Manage chat message"""
        try:
            output_generator = self.client.agent.achat(
                message.text, str(self.client.uid)
            )
            async for token in output_generator:
                logger.info(
                    f"Received token for client {self.client.get_uid()}: {token}"
                )
                await self.client.send_message("chat", token)

            logger.info(f"Stream completed for client {self.client.get_uid()}")

        # Manage exceptions
        except asyncio.CancelledError:
            logger.info(
                f"Stream cancelled for client {self.client.get_uid()} for websocket {self.client.websocket.remote_address}"
            )
            raise
        except Exception as e:
            logger.error(f"Error during chat stream: {e}")
            await self.client.send_error(
                500, f"Error during chat stream in thread {self.client.uid}: {e}"
            )

    async def message_gesture(self, message):
        """Manage gesture message"""
        pass

    async def message_image(self, message):
        """Manage image message"""
        if message.data:
            with open("src/server/test/image_received.png", "wb") as f:
                f.write(message.data)

    async def message_speech(self, message):
        """Manage json message"""
        os.makedirs("media/temp_audio", exist_ok=True)
        temp_audio_filename = f"media/temp_audio/temp_audio_{uuid.uuid4().hex}.wav"

        with open(temp_audio_filename, "wb") as f:
            f.write(message)

        text = speech_to_text(temp_audio_filename)
        await self.client.send_message("convert_speech", text)

    async def message_error(self, message):
        """Manage json message"""
        pass
