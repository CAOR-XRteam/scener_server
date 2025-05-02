from library import sql
from loguru import logger
from colorama import Fore, Style
import threading
import os

class DB:
    def __init__(self, path):
        self.path = path
        self.local = threading.local()
        self._check_path_and_init_db()

    def _check_path_and_init_db(self):
        if not os.path.exists(self.path):
            logger.warning("Database file not found.")

        conn = self._get_connection()
        cursor = self._get_cursor()
        sql.create_table_asset(conn, cursor)
        logger.success(f"Connected to database {Fore.GREEN}{self.path}{Fore.RESET}")

    def _get_connection(self):
        if not hasattr(self.local, "conn"):
            self.local.conn = sql.connect_db(self.path)
        return self.local.conn

    def _get_cursor(self):
        if not hasattr(self.local, "cursor"):
            self.local.cursor = sql.get_cursor(self._get_connection())
        return self.local.cursor

    def close(self):
        if hasattr(self.local, "conn"):
            sql.close_connection(self.local.conn)
