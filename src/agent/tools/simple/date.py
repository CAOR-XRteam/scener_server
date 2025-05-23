import datetime

from colorama import Fore
from langchain_core.tools import tool
from lib import logger


@tool
def date() -> str:
    """Returns the current date and time. No input expected."""
    logger.info(f"Using tool {Fore.GREEN}{'date'}{Fore.RESET}")
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
