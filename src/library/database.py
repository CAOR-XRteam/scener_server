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

    def fill(self, path):
        for subfolder_name in os.listdir(path):
            subpath = os.path.join(path, subfolder_name)
            if os.path.isdir(subpath):
                image = mesh = description = None
                for file_name in os.listdir(subpath):
                    file_path = os.path.join(subpath, file_name)
                    absolute_file_path = os.path.abspath(file_path)  # Get absolute path

                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image = absolute_file_path
                    elif file_name.lower().endswith(('.obj', '.fbx', '.stl', '.ply', '.glb')):
                        mesh = absolute_file_path
                    elif file_name.lower().endswith('.txt'):
                        description = absolute_file_path

                sql.insert_asset(self._get_connection(), self._get_cursor(), subfolder_name, image, mesh, description)
                logger.info(f"Inserted asset: {Fore.GREEN}{subfolder_name}{Fore.RESET}")

    def read(self):
        assets = sql.query_assets(self._get_cursor())
        if assets:
            print(f"{'ID':<4} {'Name':<10} {'Image':<10} {'Mesh':<10} {'Description':<10}")
            for asset in assets:
                asset_id, asset_name, asset_image, asset_mesh, asset_description = asset
                name = f"{Fore.YELLOW}{asset_name:<10}{Fore.RESET}"
                img = f"{Fore.GREEN}{'ok':<10}{Fore.RESET}" if asset_image else f"{Fore.RED}{'None':<10}{Fore.RESET}"
                mesh = f"{Fore.GREEN}{'ok':<10}{Fore.RESET}" if asset_mesh else f"{Fore.RED}{'None':<10}{Fore.RESET}"
                desc = f"{Fore.GREEN}{'ok':<10}{Fore.RESET}" if asset_description else f"{Fore.RED}{'None':<10}{Fore.RESET}"
                print(f"{asset_id:<4} {name} {img} {mesh} {desc}")
        else:
            print("No assets found.")

    def get_list(self):
        assets = sql.query_assets(self._get_cursor())
        return [{
            "id": asset_id,
            "name": name,
            "image": image,
            "mesh": mesh,
            "description": description
        } for asset_id, name, image, mesh, description in assets]

    def close(self):
        if hasattr(self.local, "conn"):
            sql.close_connection(self.local.conn)
