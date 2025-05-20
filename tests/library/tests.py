import library
import pytest
import sqlite3
import os

from colorama import Fore
from library.sql import Sql
from library.library_database import Database
from unittest.mock import MagicMock, patch, call


@pytest.fixture
def mock_conn():
    with patch("sqlite3.connect", spec=sqlite3.Connection) as mock_conn:
        yield mock_conn


@pytest.fixture
def mock_cursor():
    with patch("sqlite3.Cursor", spec=sqlite3.Cursor) as mock_cursor:
        yield mock_cursor


@pytest.fixture
def mock_os():
    with patch("library.library_database.os", spec=os) as mock_os:
        yield mock_os


@pytest.fixture
def mock_sql():
    with patch("library.library_database.Sql", spec=library.sql.Sql) as mock_sql:
        yield mock_sql


DB_PATH = "test.db"


@pytest.fixture
def mock_db(mock_sql, mock_os, mock_conn, mock_cursor):
    mock_os.path.exists.return_value = True
    mock_sql.connect_db.return_value = mock_conn
    mock_sql.get_cursor.return_value = mock_cursor
    with patch("library.library_database.logger"):
        db = Database(DB_PATH)
        # db._is_opened_connection = MagicMock(return_value=True)
        return db


class TestSql:
    @pytest.fixture()
    def mock_logger(self):
        with patch("library.sql.logger") as mock_logger:
            yield mock_logger

    def test_connect_db_success(self, mock_conn, mock_logger):
        db_name = "test.db"

        Sql.connect_db(db_name)

        mock_conn.assert_called_once_with(db_name)

        mock_logger.info.assert_called_once_with(
            f"Connected to the database {db_name}."
        )

    @patch("sqlite3.connect", side_effect=sqlite3.Error("test"))
    def test_connect_db_failure(
        self,
        mock_conn,
        mock_logger,
    ):
        db_name = "test.db"
        err = sqlite3.Error("test")

        with pytest.raises(sqlite3.Error, match="test"):
            Sql.connect_db(db_name)

        mock_conn.assert_called_once_with(db_name)

        mock_logger.error.assert_called_once_with(
            f"Failed to connect to the database {db_name}: {err}"
        )

    def test_get_cursor_success(self, mock_conn):
        mock_conn.cursor.return_value = MagicMock()

        Sql.get_cursor(mock_conn)

        mock_conn.cursor.assert_called_once()

    def test_get_cursor_failure(self, mock_conn, mock_logger):
        err = sqlite3.Error("test")
        mock_conn.cursor.side_effect = err

        with pytest.raises(sqlite3.Error, match="test"):
            Sql.get_cursor(mock_conn)

        mock_conn.cursor.assert_called_once()

        mock_logger.error.assert_called_once_with(
            f"Failed to get a cursor from the connection {mock_conn}: {err}"
        )

    def test_create_table_asset_success(self, mock_conn, mock_cursor, mock_logger):
        Sql.create_table_asset(mock_conn, mock_cursor)

        mock_cursor.execute.assert_called_once_with(
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
        mock_conn.commit.assert_called_once()

        mock_logger.info.assert_any_call("Created the 'asset' table.")

    def test_create_table_asset_failure_execute(
        self, mock_conn, mock_cursor, mock_logger
    ):
        err = sqlite3.Error("test")
        mock_cursor.execute.side_effect = err

        with pytest.raises(sqlite3.Error, match="test"):
            Sql.create_table_asset(mock_conn, mock_cursor)

        mock_conn.rollback.assert_called_once()

        mock_logger.error.assert_any_call(f"Failed to create the 'asset' table: {err}")

    def test_create_table_asset_failure_rollback(
        self, mock_conn, mock_cursor, mock_logger
    ):
        exec_err = sqlite3.Error("oups")
        rollback_err = sqlite3.Error("OUPSSSSS")
        mock_cursor.execute.side_effect = exec_err
        mock_conn.rollback.side_effect = rollback_err

        with pytest.raises(sqlite3.Error, match="OUPSSSSS"):
            Sql.create_table_asset(mock_conn, mock_cursor)

        mock_conn.rollback.assert_called_once()

        mock_logger.error.assert_any_call(
            f"Failed to create the 'asset' table: {exec_err}"
        )
        mock_logger.critical.assert_called_once_with(
            f"Failed to rollback: {rollback_err}"
        )

    def test_insert_asset_success_new_name(self, mock_conn, mock_cursor, mock_logger):
        mock_cursor.fetchone.return_value = (0,)

        Sql.insert_asset(
            mock_conn, mock_cursor, "asset", "img.png", "mesh.obj", "desc.txt"
        )

        calls = [
            call("SELECT COUNT(*) FROM asset WHERE name ILIKE ?", ("asset",)),
            call(
                "INSERT INTO asset (name, image, mesh, description) VALUES (?, ?, ?, ?)",
                ("asset", "img.png", "mesh.obj", "desc.txt"),
            ),
        ]

        assert mock_cursor.execute.call_count == 2
        mock_cursor.execute.assert_has_calls(calls)

        mock_conn.commit.assert_called_once()

        mock_logger.info.assert_any_call("Inserted asset 'asset' into the database.")

    def test_insert_asset_success_existing_name(
        self, mock_conn, mock_cursor, mock_logger
    ):
        mock_cursor.fetchone.return_value = (1,)

        Sql.insert_asset(
            mock_conn, mock_cursor, "asset", "img.png", "mesh.obj", "desc.txt"
        )

        calls = [
            call("SELECT COUNT(*) FROM asset WHERE name ILIKE ?", ("asset",)),
            call(
                "INSERT INTO asset (name, image, mesh, description) VALUES (?, ?, ?, ?)",
                ("asset_1", "img.png", "mesh.obj", "desc.txt"),
            ),
        ]

        assert mock_cursor.execute.call_count == 2
        mock_cursor.execute.assert_has_calls(calls)

        mock_conn.commit.assert_called_once()

        assert mock_cursor.execute.call_count == 2
        mock_logger.info.assert_any_call(
            "Asset name already exists. Inserting as 'asset_1' instead."
        )
        mock_logger.info.assert_any_call("Inserted asset 'asset_1' into the database.")

    def test_insert_asset_empty_name(self, mock_conn, mock_cursor, mock_logger):
        with pytest.raises(ValueError, match="Asset name cannot be empty"):
            Sql.insert_asset(
                mock_conn, mock_cursor, "", "img.png", "mesh.obj", "desc.txt"
            )

        mock_logger.error.assert_called_once_with(
            "Trying to insert an asset with an empty name"
        )

    def test_insert_asset_select_error(self, mock_conn, mock_cursor, mock_logger):
        err = sqlite3.Error("test")
        mock_cursor.execute.side_effect = err

        with pytest.raises(sqlite3.Error, match="test"):
            Sql.insert_asset(
                mock_conn, mock_cursor, "asset", "img.png", "mesh.obj", "desc.txt"
            )

        mock_conn.commit.assert_not_called()
        mock_conn.rollback.assert_not_called()

        mock_cursor.execute.assert_called_once_with(
            "SELECT COUNT(*) FROM asset WHERE name ILIKE ?", ("asset",)
        )

        mock_logger.error.assert_called_once_with(
            f"Failed to SELECT from 'asset' table: {err}"
        )

    def test_insert_asset_insert_error(self, mock_conn, mock_cursor, mock_logger):
        def execute_side_effect(query, params):
            if query.startswith("INSERT"):
                raise err
            else:
                return MagicMock()

        err = sqlite3.Error("test")
        mock_cursor.execute.side_effect = err
        mock_cursor.execute.side_effect = execute_side_effect
        mock_cursor.fetchone.return_value = (0,)

        with pytest.raises(sqlite3.Error, match="test"):
            Sql.insert_asset(
                mock_conn, mock_cursor, "asset", "img.png", "mesh.obj", "desc.txt"
            )

        calls = [
            call("SELECT COUNT(*) FROM asset WHERE name ILIKE ?", ("asset",)),
            call(
                "INSERT INTO asset (name, image, mesh, description) VALUES (?, ?, ?, ?)",
                ("asset", "img.png", "mesh.obj", "desc.txt"),
            ),
        ]

        assert mock_cursor.execute.call_count == 2
        mock_cursor.execute.assert_has_calls(calls)

        mock_conn.rollback.assert_called_once()

        mock_logger.error.assert_called_once_with(
            f"Failed to INSERT into 'asset' table: {err}"
        )

    def test_insert_asset_insert_rollback_error(
        self, mock_conn, mock_cursor, mock_logger
    ):
        def execute_side_effect(query, params):
            if query.startswith("INSERT"):
                raise insert_err
            else:
                return MagicMock()

        insert_err = sqlite3.Error("bim")
        rollback_err = sqlite3.Error("bim bam la sauce")
        mock_cursor.execute.side_effect = execute_side_effect
        mock_conn.rollback.side_effect = rollback_err
        mock_cursor.fetchone.return_value = (0,)

        with pytest.raises(sqlite3.Error, match="bim bam la sauce"):
            Sql.insert_asset(
                mock_conn, mock_cursor, "asset", "img.png", "mesh.obj", "desc.txt"
            )

        calls = [
            call("SELECT COUNT(*) FROM asset WHERE name ILIKE ?", ("asset",)),
            call(
                "INSERT INTO asset (name, image, mesh, description) VALUES (?, ?, ?, ?)",
                ("asset", "img.png", "mesh.obj", "desc.txt"),
            ),
        ]

        assert mock_cursor.execute.call_count == 2
        mock_cursor.execute.assert_has_calls(calls)

        mock_conn.rollback.assert_called_once()

        mock_logger.error.assert_called_once_with(
            f"Failed to INSERT into 'asset' table: {insert_err}"
        )
        mock_logger.critical.assert_called_once_with(
            f"Failed to rollback: {rollback_err}"
        )

    def test_query_assets_success(self, mock_cursor, mock_logger):
        expected_assets = [("test1",), ("test2",)]
        mock_cursor.fetchall.return_value = expected_assets

        assets = Sql.query_assets(mock_cursor)

        mock_cursor.execute.assert_called_once_with("SELECT * FROM asset")

        assert assets == expected_assets

    def test_query_assets_error(self, mock_cursor, mock_logger):
        error = sqlite3.Error("sadlife")

        mock_cursor.execute.side_effect = error

        with pytest.raises(sqlite3.Error, match="sadlife"):
            Sql.query_assets(mock_cursor)

        mock_logger.error.assert_any_call("Failed to SELECT from 'asset' table")

    def test_update_asset_success_all(self, mock_conn, mock_cursor, mock_logger):
        Sql.update_asset(
            mock_conn,
            mock_cursor,
            "asset",
            "img.png",
            "mesh.obj",
            "desc.txt",
        )

        expected_query = (
            "UPDATE asset SET image = ?, mesh = ?, description = ? WHERE name = ?"
        )
        expected_values = ("img.png", "mesh.obj", "desc.txt", "asset")

        mock_cursor.execute.assert_called_once_with(expected_query, expected_values)

        mock_conn.commit.assert_called_once()

        mock_logger.info.assert_called_once_with(
            "Updated asset 'asset' in the database."
        )

    def test_update_asset_success_partial(self, mock_conn, mock_cursor, mock_logger):
        Sql.update_asset(
            mock_conn,
            mock_cursor,
            "asset",
            image="new_img.png",
            description="new_desc.txt",
        )

        expected_query = "UPDATE asset SET image = ?, description = ? WHERE name = ?"
        expected_values = ("new_img.png", "new_desc.txt", "asset")

        mock_cursor.execute.assert_called_once_with(expected_query, expected_values)
        mock_conn.commit.assert_called_once()

        mock_logger.info.assert_called_once_with(
            "Updated asset 'asset' in the database."
        )

    def test_update_asset_no_fields_to_update(
        self, mock_conn, mock_cursor, mock_logger
    ):
        with pytest.raises(
            ValueError, match="No fields to update provided to the asset '{name}'"
        ):
            Sql.update_asset(mock_conn, mock_cursor, "asset")

        mock_cursor.execute.assert_not_called()

        mock_conn.commit.assert_not_called()

        mock_logger.info.assert_not_called()

    def test_update_asset_error(self, mock_conn, mock_cursor, mock_logger):
        err = sqlite3.Error("zzz")
        mock_cursor.execute.side_effect = err

        with pytest.raises(sqlite3.Error, match="zzz"):
            Sql.update_asset(mock_conn, mock_cursor, "asset", image="new_img.png")

        mock_conn.rollback.assert_called_once()

        mock_logger.error.assert_called_once_with(
            f"Faield to UPDATE the 'asset' table: {err}"
        )

    def test_update_asset_rollback_error(self, mock_conn, mock_cursor, mock_logger):
        update_err = sqlite3.Error("sad")
        rollback_err = sqlite3.Error("bad")
        mock_cursor.execute.side_effect = update_err
        mock_conn.rollback.side_effect = rollback_err

        with pytest.raises(sqlite3.Error, match="bad"):
            Sql.update_asset(mock_conn, mock_cursor, "asset", image="new_img.png")

        mock_conn.rollback.assert_called_once()

        mock_logger.error.assert_called_once_with(
            f"Faield to UPDATE the 'asset' table: {update_err}"
        )
        mock_logger.critical.assert_called_once_with(
            f"Failed to rollback: {rollback_err}"
        )

    def test_delete_asset_success(self, mock_conn, mock_cursor, mock_logger):
        Sql.delete_asset(mock_conn, mock_cursor, "asset")

        mock_cursor.execute.assert_called_once_with(
            "DELETE FROM asset WHERE name = ?", ("asset",)
        )

        mock_logger.info.assert_called_once_with(
            "Deleted asset 'asset' from the database."
        )

        mock_conn.commit.assert_called_once()

    def test_delete_asset_error(self, mock_conn, mock_cursor, mock_logger):
        err = sqlite3.Error("Delete failed")
        mock_cursor.execute.side_effect = err

        with pytest.raises(sqlite3.Error, match="Delete failed"):
            Sql.delete_asset(mock_conn, mock_cursor, "asset")

        mock_conn.rollback.assert_called_once()

        mock_logger.error.assert_called_once_with(
            f"Failed DELETE from 'asset' table: {err}"
        )

    def test_delete_asset_rollback_error(self, mock_conn, mock_cursor, mock_logger):
        delete_err = sqlite3.Error("bim")
        rollback_err = sqlite3.Error("BAM")
        mock_cursor.execute.side_effect = delete_err
        mock_conn.rollback.side_effect = rollback_err

        with pytest.raises(sqlite3.Error, match="BAM"):
            Sql.delete_asset(mock_conn, mock_cursor, "asset")

        mock_conn.rollback.assert_called_once()

        mock_logger.error.assert_called_once_with(
            f"Failed DELETE from 'asset' table: {delete_err}"
        )
        mock_logger.critical.assert_called_once_with(
            f"Failed to rollback: {rollback_err}"
        )

    def test_close_connection_success(self, mock_conn, mock_logger):
        Sql.close_connection(mock_conn)

        mock_conn.close.assert_called_once()

        mock_logger.info.assert_called_once_with(
            f"Closed the database connection {mock_conn}."
        )

    def test_close_connection_error(self, mock_conn, mock_logger):
        err = sqlite3.Error("aoaiaioa")
        mock_conn.close.side_effect = err

        with pytest.raises(sqlite3.Error, match="aoaiaioa"):
            Sql.close_connection(mock_conn)

        mock_logger.error.assert_called_once_with(
            f"Failed to close the {mock_conn} connection: {err}"
        )

    def test_retry(self, mock_logger, mock_conn, mock_cursor, monkeypatch):
        err = sqlite3.OperationalError("Database is locked")
        mock_cursor.execute.side_effect = err

        func = Sql.create_table_asset
        monkeypatch.setattr(func.retry, "before_sleep", mock_logger.error)
        monkeypatch.setattr(func.retry, "after", mock_logger.info)
        monkeypatch.setattr(func.retry, "wait", 0)

        with pytest.raises(sqlite3.OperationalError, match="Database is locked"):
            Sql.create_table_asset(mock_conn, mock_cursor)

        assert mock_cursor.execute.call_count == 4

        assert mock_conn.rollback.call_count == 4

        assert mock_logger.error.call_count == 7
        assert mock_logger.info.call_count == 4


class TestDatabase:

    @pytest.fixture
    def mock_logger(self):
        with patch("library.library_database.logger") as mock_logger:
            yield mock_logger

    def test_init_success_db_exists(
        self, mock_sql, mock_logger, mock_os, mock_conn, mock_cursor
    ):
        mock_os.path.exists.return_value = True
        mock_sql.connect_db.return_value = mock_conn
        mock_sql.get_cursor.return_value = mock_cursor

        db = Database(DB_PATH)

        assert db.path == DB_PATH
        assert db._conn == mock_conn

        mock_os.path.exists.assert_called_once_with(DB_PATH)
        mock_os.makedirs.assert_called_once_with(
            mock_os.path.dirname(DB_PATH), exist_ok=True
        )

        mock_sql.connect_db.assert_called_once_with(DB_PATH)
        mock_sql.get_cursor.assert_called_once_with(db._conn)
        mock_sql.create_table_asset.assert_called_once_with(mock_conn, mock_cursor)

        mock_logger.info.assert_any_call(
            f"Database initialized at {Fore.GREEN}{DB_PATH}{Fore.RESET}"
        )
        mock_logger.success.assert_called_once_with(
            f"Connected to database {Fore.GREEN}{DB_PATH}{Fore.RESET}"
        )

    def test_init_success_db_not_exists(
        self, mock_sql, mock_logger, mock_os, mock_conn, mock_cursor
    ):
        mock_os.path.exists.return_value = False
        mock_sql.connect_db.return_value = mock_conn
        mock_sql.get_cursor.return_value = mock_cursor

        db = Database(DB_PATH)

        assert db.path == DB_PATH
        assert db._conn == mock_conn

        mock_os.path.exists.assert_called_once_with(DB_PATH)
        mock_os.makedirs.assert_called_once_with(
            mock_os.path.dirname(DB_PATH), exist_ok=True
        )

        mock_sql.connect_db.assert_called_once_with(DB_PATH)
        mock_sql.get_cursor.assert_called_once_with(db._conn)
        mock_sql.create_table_asset.assert_called_once_with(mock_conn, mock_cursor)

        mock_logger.info.assert_any_call(
            f"Database initialized at {Fore.GREEN}{DB_PATH}{Fore.RESET}"
        )
        mock_logger.success.assert_called_once_with(
            f"Connected to database {Fore.GREEN}{DB_PATH}{Fore.RESET}"
        )
        mock_logger.warning.assert_called_once_with(
            f"Database file not found. Creating it at {Fore.GREEN}{DB_PATH}{Fore.RESET}."
        )

    def test_init_makedir_error(self, mock_logger, mock_os):
        err = OSError("euh")
        mock_os.path.exists.return_value = False
        mock_os.makedirs.side_effect = err

        with pytest.raises(OSError, match="euh"):
            _ = Database(DB_PATH)

        mock_logger.warning.assert_called_once_with(
            f"Database file not found. Creating it at {Fore.GREEN}{DB_PATH}{Fore.RESET}."
        )
        assert mock_logger.error.call_count == 2
        mock_logger.error.assert_any_call(
            f"Failed to create directory for database: {err}"
        )
        mock_logger.error.assert_any_call(f"Failed to initialize database: {err}")

    def test_init_connect_error(self, mock_sql, mock_logger, mock_os):
        err = sqlite3.Error("ah")
        mock_os.path.exists.return_value = True
        mock_sql.connect_db.side_effect = err

        with pytest.raises(sqlite3.Error, match="ah"):
            db = Database(DB_PATH)
            assert db._conn is None

        assert mock_logger.error.call_count == 2
        mock_logger.error.assert_any_call(f"Failed to initialize database: {err}")

    def test_init_get_cursor_error_close_conn_success(
        self, mock_sql, mock_logger, mock_os, mock_conn
    ):
        err = sqlite3.Error("oh")
        mock_os.path.exists.return_value = True
        mock_sql.connect_db.return_value = mock_conn
        mock_sql.get_cursor.side_effect = err

        with pytest.raises(sqlite3.Error, match="oh"):
            db = Database(DB_PATH)
            assert db._conn is None

        mock_sql.connect_db.assert_called_once_with(DB_PATH)
        mock_sql.close_connection.assert_called_once_with(mock_conn)

        assert mock_logger.error.call_count == 2
        mock_logger.error.assert_any_call(f"Failed to initialize database: {err}")

    def test_init_get_cursor_error_close_conn_error(
        self, mock_sql, mock_logger, mock_os, mock_conn
    ):
        get_cursor_err = sqlite3.Error("ba")
        close_err = sqlite3.Error("daboum")
        mock_os.path.exists.return_value = True
        mock_sql.connect_db.return_value = mock_conn
        mock_sql.get_cursor.side_effect = get_cursor_err
        mock_sql.close_connection.side_effect = close_err

        with pytest.raises(sqlite3.Error, match="daboum"):
            db = Database(DB_PATH)
            assert db._conn is None

        mock_sql.connect_db.assert_called_once_with(DB_PATH)
        mock_sql.get_cursor.assert_called_once_with(mock_conn)
        mock_sql.close_connection.assert_called_once_with(mock_conn)

        assert mock_logger.error.call_count == 3
        mock_logger.error.assert_any_call(
            f"Failed to initialize database: {get_cursor_err}"
        )
        mock_logger.error.assert_any_call(f"Failed to close connection: {close_err}")

    def test_init_create_asset_error(
        self, mock_sql, mock_logger, mock_os, mock_conn, mock_cursor
    ):
        err = sqlite3.Error("pims sont sous-cotés")
        mock_os.path.exists.return_value = True
        mock_sql.connect_db.return_value = mock_conn
        mock_sql.get_cursor.return_value = mock_cursor
        mock_sql.create_table_asset.side_effect = err

        with pytest.raises(sqlite3.Error, match="pims sont sous-coté"):
            db = Database(DB_PATH)
            assert db._conn is None

        mock_sql.connect_db.assert_called_once_with(DB_PATH)
        mock_sql.get_cursor.assert_called_once_with(mock_conn)
        mock_sql.close_connection.assert_called_once_with(mock_conn)

        assert mock_logger.error.call_count == 2
        mock_logger.error.assert_any_call(f"Failed to initialize database: {err}")

    def test_get_conn_already_opened(self, mock_db, mock_sql, mock_conn):
        conn = mock_db.get_connection()

        assert conn == mock_conn

        mock_sql.get_cursor.assert_called_once_with(mock_conn)

        assert mock_db._is_opened_connection() is True

    def test_get_conn_new_conn_success(self, mock_db, mock_sql, mock_conn):
        mock_db._conn = None
        mock_sql.get_cursor.side_effect = sqlite3.Error("test")

        assert mock_db._is_opened_connection() is False

        conn = mock_db.get_connection()

        assert conn == mock_conn

        mock_sql.get_cursor.assert_called_once()

    def test_get_conn_new_conn_error(self, mock_db, mock_sql, mock_logger):
        err = sqlite3.Error("test")
        mock_db._conn = None
        mock_sql.connect_db.side_effect = err
        mock_sql.get_cursor.side_effect = err

        assert mock_db._is_opened_connection() is False

        with pytest.raises(sqlite3.Error, match="test"):
            mock_db.get_connection()

        mock_sql.get_cursor.assert_called_once()

        mock_logger.error.assert_called_once_with(
            f"Failed to create a new connection: {err}"
        )

    def test_get_cursor_success(self, mock_db, mock_sql, mock_conn, mock_cursor):
        cursor = mock_db._get_cursor()

        assert cursor == mock_cursor

        assert mock_sql.get_cursor.call_count == 2
        mock_sql.get_cursor.assert_any_call(mock_conn)

    def test_get_cursor_error(self, mock_db, mock_sql, mock_conn, mock_logger):
        err = sqlite3.Error("test")
        mock_sql.get_cursor.side_effect = err

        with pytest.raises(sqlite3.Error, match="test"):
            mock_db._get_cursor()

        assert mock_sql.get_cursor.call_count == 2
        mock_sql.get_cursor.assert_any_call(mock_conn)

        mock_logger.error.assert_called_once_with(f"Failed to get a cursor: {err}")

    def test_close_connection_no_opened_connecion(self, mock_db, mock_logger):
        mock_db._conn = None
        mock_db.close()

        mock_logger.warning.assert_called_once_with("No connection to close.")

    def test_close_connection_success(self, mock_db, mock_sql):
        mock_db.close(mock_db._conn)

        mock_sql.close_connection.assert_called_once_with(mock_db._conn)

    def test_close_connection_error(self, mock_db, mock_sql, mock_logger):
        err = sqlite3.Error("test")
        mock_sql.close_connection.side_effect = err

        with pytest.raises(sqlite3.Error, match="test"):
            mock_db.close(mock_db._conn)

        mock_sql.close_connection.assert_called_once_with(mock_db._conn)

        mock_logger.error.assert_called_once_with(f"Failed to close connection: {err}")


class TestAsset:
    pass


class TestLibrary:
    pass
