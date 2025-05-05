import utils
import library
import server
import model
from src.lib import setup_logging

if __name__ == "__main__":
    utils.init()
    library.init()
    server.start()
    setup_logging()
