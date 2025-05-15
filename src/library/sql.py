"""
sql.py

Low-level SQL functions

Author: Nathan SV
Created: 05-05-2025
Last Updated: 05-05-2025
"""

import sqlite3

from beartype import beartype
from lib import logger


@beartype
class Sql:
    # Init
    def connect_db(self, db_name: str):
        """Connect to an SQLite database (create it if not exists) and return the connection."""
        try:
            conn = sqlite3.connect(db_name)
            logger.info(f"Connected to the database {db_name}.")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Couldn't connect to the database: {e}")
            raise

    def get_cursor(conn: sqlite3.Connection):
        """Return a cursor from the connection."""
        try:
            return conn.cursor()
        except sqlite3.Error as e:
            logger.error(f"Couldn't get a cursor from the connection {conn}: {e}")
            raise

    # Operation
    def create_table_asset(self, conn: sqlite3.Connection, cursor: sqlite3.Cursor):
        """Create an 'assets' table if it doesn't exist."""
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
        except sqlite3.Error as e:
            logger.error(f"Couldn't create an 'assets' table: {e}")
            try:
                conn.rollback()
            except sqlite3.Error as e:
                logger.error(f"Couldn't rollback: {e}")
            finally:
                raise

    def insert_asset(
        self,
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
            raise

        # Check if asset with the same name already exists rename if it does
        try:
            cursor.execute("SELECT COUNT(*) FROM asset WHERE name ILIKE ?", (name,))
            nb = cursor.fetchone()[0]
            if nb > 0:
                name = name + f"_{nb+1}"
        except sqlite3.Error as e:
            logger.error(f"Couldn't SELECT from asset table: {e}")

        try:
            cursor.execute(
                "INSERT INTO asset (name, image, mesh, description) VALUES (?, ?, ?, ?)",
                (name, image, mesh, description),
            )
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Couldn't INSERT into asset table: {e}")
            try:
                conn.rollback()
            except sqlite3.Error as e:
                logger.error(f"Error while rollbacking: {e}")
            finally:
                raise

    def query_assets(self, cursor: sqlite3.Cursor):
        """Fetch all assets from the 'asset' table."""
        try:
            cursor.execute("SELECT * FROM asset")
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error("Couldn't SELECT from asset table")
            raise

    def update_asset(
        self,
        conn: sqlite3.Connection,
        cursor: sqlite3.Connection,
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
        except sqlite3.Error as e:
            logger.error(f"Couldn't UPDATE asset table: {e}")
            try:
                conn.rollback()
            except sqlite3.Error as e:
                logger.error(f"Error while rollbacking: {e}")
            finally:
                raise

    def delete_asset(self, conn: sqlite3.Connection, cursor: sqlite3.Cursor, name: str):
        """Delete an asset by its name."""
        try:
            cursor.execute("DELETE FROM asset WHERE name = ?", (name,))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Couldn't DELETE from asset table: {e}")
            try:
                conn.rollback()
            except sqlite3.Error as e:
                logger.error(f"Error while rollbacking: {e}")
            finally:
                raise

    # Closing
    def close_connection(self, conn: sqlite3.Connection):
        """Close the SQLite connection."""
        try:
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Couldn't close the {conn} connection: {e}")
            raise
