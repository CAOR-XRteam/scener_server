"""
library_database.py

Functions to connect to the sql database

Author: Nathan SV
Created: 05-05-2025
Last Updated: 19-05-2025
"""

# TODO: more precise error handling to propagate to the agent

import os
import sqlite3

from beartype import beartype
from colorama import Fore
from library.sql import Sql
from loguru import logger


@beartype
class Database:
    def _is_opened_connection(self):
        # Check if the connection is open
        try:
            self._conn.cursor()
            return True
        except Exception as e:
            return False

    def _check_path_and_init_db(self):
        # Ensure that the database is properly initialized
        if not os.path.exists(self.path):
            logger.warning(
                f"Database file not found. Creating it at {Fore.GREEN}{self.path}{Fore.RESET}."
            )

        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory for database: {e}")
            raise

        try:
            self._conn = Sql.connect_db(self.path)
            cursor = Sql.get_cursor(self._conn)
            Sql.create_table_asset(self._conn, cursor)
            logger.success(f"Connected to database {Fore.GREEN}{self.path}{Fore.RESET}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            if self._conn:
                try:
                    Sql.close_connection(self._conn)
                except Exception as close_e:
                    logger.error(f"Failed to close connection: {close_e}")
                    raise
                finally:
                    self._conn = None
            raise

    def __init__(self, path: str):
        self.path = path
        self._conn = None
        try:
            self._check_path_and_init_db()
            logger.info(f"Database initialized at {Fore.GREEN}{self.path}{Fore.RESET}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def get_connection(self):
        # Create a new connection each time it's needed, no need to cache
        if self._is_opened_connection():
            return self._conn
        else:
            try:
                self._conn = Sql.connect_db(self.path)
                return self._conn
            except Exception as e:
                self._conn = None
                logger.error(f"Failed to create a new connection: {e}")
                raise

    def _get_cursor(self):
        # Get a fresh cursor for each operation
        try:
            conn = self.get_connection()
            return Sql.get_cursor(conn)
        except Exception as e:
            logger.error(f"Failed to get a cursor: {e}")
            raise

    def close(self, conn: sqlite3.Connection = None):
        # Close the specific connection
        if conn:
            try:
                Sql.close_connection(conn)
            except Exception as e:
                logger.error(f"Failed to close connection: {e}")
                raise
        else:
            logger.warning("No connection to close.")
