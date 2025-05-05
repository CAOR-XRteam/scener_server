"""
sql.py

Low-level SQL functions

Author: Nathan SV
Created: 05-05-2025
Last Updated: 05-05-2025
"""


import sqlite3

# Init
def connect_db(db_name):
    """Connect to an SQLite database (create it if not exists) and return the connection."""
    return sqlite3.connect(db_name)

def get_cursor(conn):
    """Return a cursor from the connection."""
    return conn.cursor()

# Operation
def create_table_asset(conn, cursor):
    """Create an 'assets' table if it doesn't exist."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS asset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image TEXT,
            mesh TEXT,
            description TEXT
        )
    ''')
    conn.commit()

def insert_asset(conn, cursor, name, image, mesh, description):
    """Insert a new asset into the 'asset' table if the name does not already exist."""
    if not name:
        return

    # Check if asset with the same name already exists
    cursor.execute("SELECT COUNT(*) FROM asset WHERE name = ?", (name,))
    if cursor.fetchone()[0] > 0:
        return

    # Insert the new asset
    cursor.execute("INSERT INTO asset (name, image, mesh, description) VALUES (?, ?, ?, ?)", (name, image, mesh, description))
    conn.commit()

def query_assets(cursor):
    """Fetch all assets from the 'asset' table."""
    cursor.execute("SELECT * FROM asset")
    return cursor.fetchall()

def update_asset(conn, cursor, name, image=None, mesh=None, description=None):
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
    cursor.execute(f"UPDATE asset SET {update_fields_str} WHERE name = ?", tuple(update_values))
    conn.commit()

def delete_asset(conn, cursor, name):
    """Delete an asset by its name."""
    cursor.execute("DELETE FROM asset WHERE name = ?", (name,))
    conn.commit()

# Closing
def close_connection(conn):
    """Close the SQLite connection."""
    conn.close()
