import json
import logging
import uuid

from beartype import beartype
from agent.api import AgentAPI
from server.client import Client
from loguru import logger


@beartype
class Session:
    def __init__(self, client: Client, model_name: str = "llama3.1"):
        self.client = client
        self.thread_id = uuid.uuid1()
        self.model_name = model_name

        try:
            self.agent = AgentAPI(model_name)
            logger.info(
                f"Session created with thread_id: {self.thread_id} for websocket {self.client.websocket.remote_address} with model {self.model_name}"
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
        try:
            j = json.loads(message)
            if j.get("command") == "chat":
                input = j.get("message")
                if not input:
                    await self.client.send_message(
                        "error", 400, f"Empty message in thread {self.thread_id}"
                    )
                try:
                    async for token in self.agent.chat(input, self.thread_id):
                        await self.client.send_message("stream", 200, token)
                except Exception as e:
                    logger.error(f"Error during chat stream: {e}")
                    await self.client.send_message(
                        "error",
                        500,
                        f"Error during chat stream in thread {self.thread_id}",
                    )
            else:
                await self.client.send_message(
                    "error", 400, f"Command not recognized in thread {self.thread_id}"
                )
        except json.JSONDecodeError:
            await self.client.send_message(
                "error", 400, f"Invalid JSON format in thread {self.thread_id}"
            )
