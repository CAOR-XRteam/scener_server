import redis
import os
from lib import logger
from colorama import Fore


class Redis:
    def __init__(self):
        self.host = os.getenv("HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", 8765))
        self.client: redis.Redis | None = None
        logger.info(f"Redis initialized for {self.host}:{self.port}")

    async def connect(self):
        if self.client:
            logger.info("Redis client already connected.")
            return

        try:
            logger.info("Connecting to Redis...")
            self.client = redis.Redis(
                host=self.host, port=self.port, db=0, decode_responses=True
            )
            self.client.ping()
            logger.success(
                f"Successfully connected to Redis at {Fore.GREEN}{self.host}:{self.port}{Fore.RESET}"
            )
        except Exception as e:
            logger.critical(f"Failed to connect to Redis: {e}")
            self.client = None
            raise

    async def disconnect(self):
        if self.client:
            logger.info("Disconnecting from Redis...")
            await self.client.close()
            self.client = None
            logger.success("Redis connection closed")

    async def get_scene(self, thread_id: str) -> str | None:
        if not self.client:
            logger.error("Redis client is not connected")
            return None

        key = f"scene:{thread_id}"
        return await self.client.get(key)
