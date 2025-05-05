from src.utils import config
import library
import server
from src.lib import setup_logging

if __name__ == "__main__":
    config = config.load_config()
    try:
        host, port, model_name = (
            config.get("host"),
            config.get("localhost"),
            config.get("model_name"),
        )
    except KeyError:
        host = "localhost"
        port = 8000
    library.init()
    server.start(host, port, model_name)
    setup_logging()
