from langchain_core.tools import tool
import datetime

@tool
def date(e) -> str:
    """Returns the current date. No input expected."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
