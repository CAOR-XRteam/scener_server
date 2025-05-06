import uuid

from beartype import beartype
from agent.api import AgentAPI
from server.client import Client
from lib import logger


@beartype
class Session:
    def __init__(self, client: Client):
        self.client = client
        self.thread_id = uuid.uuid1()

        try:
            self.agent = AgentAPI()
            logger.info(
                f"Session created with thread_id: {self.thread_id} for websocket {self.client.websocket.remote_address}"
            )
        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
            self.client.is_active = False

    async def run(self):
        while self.client.is_active:
            try:
                message = await self.client.queue_input.get()
                await self.handle_message(message)
                self.client.queue_input.task_done()
            except Exception as e:
                # Optional: log or handle processing errors
                logger.error(f"Session error: {e}")
                await self.client.send_message(
                    "error", 500, f"Internal server error in thread {self.thread_id}"
                )
                break

    async def handle_message(self, message: str):
        if not message or message.isspace():
            await self.client.send_message(
                "error", 400, f"Empty message in thread {self.thread_id}"
            )
            return

        logger.info(f"Received message in thread {self.thread_id}: {message}")

        try:
            output_generator = self.agent.achat(message, self.thread_id)
            logger.info(f"{output_generator}")
            async for token in output_generator:
                await self.client.send_message("stream", 200, token)

            logger.info(f"Stream completed for thread {self.thread_id}")

        except Exception as e:
            logger.error(f"Error during chat stream: {e}")
            await self.client.send_message(
                "error",
                500,
                f"Error during chat stream in thread {self.thread_id}",
            )
