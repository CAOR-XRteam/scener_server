import json
import logging
import uuid

from beartype import beartype
from model.llm.agent import Agent
from server.client import Client

logger = logging.getLogger(__name__)


@beartype
class Session:
    def __init__(self, client: Client):
        self.client = client
        self.thread_id = uuid.uuid1()
        self.agent = Agent()
        logger.info(
            f"Session created with thread_id: {self.thread_id} for websocket {self.client.websocket.remote_address}"
        )

    async def run(self):
        while self.client.is_active:
            try:
                message = await self.client.queue_input.get()
                await self.handle_message(message)
                self.client.queue_input.task_done()
            except Exception as e:
                # Optional: log or handle processing errors
                logger.error(f"Session error: {e}")
                break

    async def handle_message(self, message: str):
        try:
            j = json.loads(message)
            if j.get("command") == "chat":
                input = j.get("message")
                if not input:
                    await self.client.send_message(
                        "error", 400, f"Empty message in thread {self.thread_id}"
                    )
                output = self.agent.chat(input, self.thread_id)
                await self.client.send_message("Success", 200, output)
            else:
                await self.client.send_message(
                    "error", 400, f"Command not recognized in thread {self.thread_id}"
                )
        except json.JSONDecodeError:
            await self.client.send_message(
                "error", 400, f"Invalid JSON format in thread {self.thread_id}"
            )
