from agent.api import AgentAPI
from server.io.valider import InputMessage, OutputMessage
from lib import logger
from beartype import beartype
from colorama import Fore
from pydantic import ValidationError
import asyncio


@beartype
class Queue:
    """Manage client queues"""

    def __init__(self):
        self.input: asyncio.Queue[InputMessage] = (asyncio.Queue())
        self.output: asyncio.Queue[OutputMessageWrapper] = (asyncio.Queue())

    def clear(self):
        """Clear queues without blocking."""
        while not self.input.empty():
            try:
                self.input.get_nowait()
                self.input.task_done()
            except asyncio.QueueEmpty:
                break
        while not self.output.empty():
            try:
                self.output.get_nowait()
                self.output.task_done()
            except asyncio.QueueEmpty:
                break
