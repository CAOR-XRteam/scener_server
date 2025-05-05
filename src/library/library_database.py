"""
library_database.py

Functions to connect to the sql database

Author: Nathan SV
Created: 05-05-2025
Last Updated: 05-05-2025
"""

from loguru import logger
from colorama import Fore, Style
import sql
import threading
import os

class Database:
    def __init__(self, path):
        self.path = path
        self._check_path_and_init_db()

    def _check_path_and_init_db(self):
        if not os.path.exists(self.path):
            logger.warning(f"Database file not found {Fore.GREEN}{self.path}{Fore.RESET}.")

        # Ensure that the database is properly initialized
        conn = self._get_connection()
        cursor = self._get_cursor()
        sql.create_table_asset(conn, cursor)
        logger.success(f"Connected to database {Fore.GREEN}{self.path}{Fore.RESET}")

    def _get_connection(self):
        # Creates a new connection each time it's needed, no need to cache
        return sql.connect_db(self.path)

    def _get_cursor(self):
        # Get a fresh cursor for each operation
        conn = self._get_connection()
        return sql.get_cursor(conn)

    def close(self, conn=None):
        # Close the specific connection
        if conn:
            sql.close_connection(conn)
        else:
            logger.warning("No connection to close.")
