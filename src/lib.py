import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("langgraph").setLevel(logging.DEBUG)
    logging.getLogger("langchain_core").setLevel(logging.DEBUG)
