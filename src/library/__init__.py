from .api import db
from loguru import logger
import sys


#Loguru config
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
