from .api import server
from loguru import logger
import sys


<<<<<<< HEAD

def start():
    global server
    if server is None:
        server = ws.Server()
    server.start()
=======
#Loguru config
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
>>>>>>> main
