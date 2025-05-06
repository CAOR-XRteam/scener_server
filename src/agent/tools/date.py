from langchain_core.tools import tool
from loguru import logger
from colorama import Fore
import datetime

@tool
def date() -> str:
    """Returns the current date and time. No input expected."""
    logger.info(f"Using tool {Fore.GREEN}{'date'}{Fore.RESET}")
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
