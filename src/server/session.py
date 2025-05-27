import asyncio
import uuid

from beartype import beartype
from server.client import Client
from lib import logger
from server.valider import InputMessage, OutputMessage


@beartype
class Session:
    def __init__(self, client: Client):
        """init session by client and assign an ID"""
        self.client = client
        self.thread_id = uuid.uuid1()

    async def run(self):
        """While client keep being actif, handle input messages"""
        while self.client.is_active:
            try:
                message = await self.client.queue_input.get()
                await self.handle_message(message)
                self.client.queue_input.task_done()
            except asyncio.CancelledError:
                logger.info(
                    f"Session {self.thread_id} cancelled for websocket {self.client.websocket.remote_address}"
                )
                break
            except Exception as e:
                # Optional: log or handle processing errors
                logger.error(f"Session error: {e}")
                await self.client.send_message(
                    OutputMessage(
                        status="error",
                        code=500,
                        message=f"Internal server error in thread {self.thread_id}",
                    )
                )
                break

    async def handle_message(self, input_message: InputMessage):
        """handle one client input message - send it to async chat"""
        message = input_message.message
        logger.info(f"Received message in thread {self.thread_id}: {message}")

        try:
            output_generator = self.client.agent.achat(message, str(self.thread_id))
            async for token in output_generator:
                await self.client.send_message(
                    OutputMessage(status="stream", code=200, message=token)
                )

            logger.info(f"Stream completed for thread {self.thread_id}")
        except asyncio.CancelledError:
            logger.info(
                f"Stream cancelled for thread {self.thread_id} for websocket {self.client.websocket.remote_address}"
            )
            raise
        except Exception as e:
            logger.error(f"Error during chat stream: {e}")
            await self.client.send_message(
                OutputMessage(
                    status="error",
                    code=500,
                    message=f"Error during chat stream in thread {self.thread_id}",
                )
            )
