from langchain.callbacks.base import BaseCallbackHandler
from loguru import logger
from colorama import Fore


""" Custom tool tracker for functionnal tests """


class Tool_callback(BaseCallbackHandler):
    def __init__(self):
        self.used_tools = []

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Start when a tool is being to be used"""
        # Log starting tool
        tool_name = serialized.get("name")
        logger.info(f"Using tool {Fore.GREEN}{tool_name}{Fore.RESET}")

        # Store used tool names
        if tool_name and tool_name not in self.used_tools:
            self.used_tools.append(tool_name)

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Stop when a tool has done"""
        pass
