"""
sql.py

Low-level SQL functions

Author: Nathan SV
Created: 05-05-2025
Last Updated: 19-05-2025
"""

import sqlite3

from beartype import beartype
from lib import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)


@beartype
class Sql:
    retry_on_db_lock = retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, min=0.1, max=2),
        retry=retry_if_exception_type(sqlite3.OperationalError),
        before_sleep=before_sleep_log(logger, "ERROR"),
        after=after_log(logger, "INFO"),
        reraise=True,
    )

    # Init
    @staticmethod
    def connect_db(db_name: str):
        """Connect to an SQLite database (create it if doesn't exist) and return the connection."""
        try:
            conn = sqlite3.connect(db_name)
            logger.info(f"Connected to the database {db_name}.")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to the database {db_name}: {e}")
            raise

    @staticmethod
    def get_cursor(conn: sqlite3.Connection):
        """Return a cursor from the connection."""
        try:
            return conn.cursor()
        except sqlite3.Error as e:
            logger.error(f"Failed to get a cursor from the connection {conn}: {e}")
            raise

    # Operation
    @staticmethod
    @retry_on_db_lock
    def create_table_asset(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
        """Create an 'asset' table if it doesn't exist."""
        try:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS asset (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    image TEXT,
                    mesh TEXT,
                    description TEXT
                )
            """
            )
            conn.commit()
            logger.info("Created the 'asset' table.")
        except sqlite3.Error as e:
            logger.error(f"Failed to create the 'asset' table: {e}")
            try:
                conn.rollback()
            except sqlite3.Error as e:
                logger.critical(f"Failed to rollback: {e}")
            finally:
                raise

    @staticmethod
    @retry_on_db_lock
    def insert_asset(
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        name: str,
        image: str,
        mesh: str,
        description: str,
    ):
        """Insert a new asset into the 'asset' table if the name does not already exist."""
        if not name:
            logger.error("Trying to insert an asset with an empty name")
            raise ValueError("Asset name cannot be empty")

        # Check if asset with the same name already exists rename if it does
        try:
            cursor.execute("SELECT COUNT(*) FROM asset WHERE name ILIKE ?", (name,))
            nb = cursor.fetchone()[0]
            if nb > 0:
                name = name + f"_{nb+1}"
                logger.info(
                    f"Asset name already exists. Inserting as '{name}' instead."
                )
        except sqlite3.Error as e:
            logger.error(f"Failed to SELECT from 'asset' table: {e}")
            raise

        try:
            cursor.execute(
                "INSERT INTO asset (name, image, mesh, description) VALUES (?, ?, ?, ?)",
                (name, image, mesh, description),
            )
            conn.commit()
            logger.info(f"Inserted asset {name} into the database.")
        except sqlite3.Error as e:
            logger.error(f"Failed to INSERT into 'asset' table: {e}")
            try:
                conn.rollback()
            except sqlite3.Error as e:
                logger.critical(f"Falied to rollback: {e}")
                raise
            raise

    @staticmethod
    @retry_on_db_lock
    def query_assets(cursor: sqlite3.Cursor):
        """Fetch all assets from the 'asset' table."""
        try:
            cursor.execute("SELECT * FROM asset")
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error("Failed to SELECT from 'asset' table")
            raise

    @staticmethod
    @retry_on_db_lock
    def update_asset(
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        name: str,
        image: str = None,
        mesh: str = None,
        description: str = None,
    ):
        """Update an existing asset's information by its name."""

        # Build the SET clause for the SQL query dynamically based on the non-None parameters
        update_fields = []
        update_values = []

        if image is not None:
            update_fields.append("image = ?")
            update_values.append(image)

        if mesh is not None:
            update_fields.append("mesh = ?")
            update_values.append(mesh)

        if description is not None:
            update_fields.append("description = ?")
            update_values.append(description)

        if not update_fields:
            return  # If no fields to update, return early

        # Add the asset name at the end of the update_values to match the WHERE clause
        update_fields_str = ", ".join(update_fields)
        update_values.append(name)

        # Execute the update query
        try:
            cursor.execute(
                f"UPDATE asset SET {update_fields_str} WHERE name = ?",
                tuple(update_values),
            )
            conn.commit()
            logger.info(f"Updated asset {name} in the database.")
        except sqlite3.Error as e:
            logger.error(f"Faield to UPDATE the 'asset' table: {e}")
            try:
                conn.rollback()
            except sqlite3.Error as e:
                logger.critical(f"Failed to rollback: {e}")
                raise
            raise

    @staticmethod
    @retry_on_db_lock
    def delete_asset(conn: sqlite3.Connection, cursor: sqlite3.Cursor, name: str):
        """Delete an asset by its name."""
        try:
            cursor.execute("DELETE FROM asset WHERE name = ?", (name,))
            conn.commit()
            logger.info(f"Deleted asset {name} from the database.")
        except sqlite3.Error as e:
            logger.error(f"Failed DELETE from asset table: {e}")
            try:
                conn.rollback()
            except sqlite3.Error as e:
                logger.critical(f"Faied to rollback: {e}")
            finally:
                raise

    # Closing
    @staticmethod
    @retry_on_db_lock
    def close_connection(conn: sqlite3.Connection):
        """Close the SQLite connection."""
        try:
            conn.close()
            logger.info(f"Closed the database connection {conn}.")
        except sqlite3.Error as e:
            logger.error(f"Failed to close the {conn} connection: {e}")
            raise
