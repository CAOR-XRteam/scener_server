from loguru import logger
from colorama import Fore
import sql
import os

class Library:
    def __init__(self, db: 'DB'):
        self.db = db

    def fill(self, path):
        """Fill the database with assets from the specified directory."""
        conn = self.db._get_connection()
        cursor = self.db._get_cursor()  # fresh cursor

        for subfolder_name in os.listdir(path):
            subpath = os.path.join(path, subfolder_name)
            if os.path.isdir(subpath):
                image = mesh = description = None
                for file_name in os.listdir(subpath):
                    file_path = os.path.join(subpath, file_name)
                    absolute_file_path = os.path.abspath(file_path)

                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image = absolute_file_path
                    elif file_name.lower().endswith(('.obj', '.fbx', '.stl', '.ply', '.glb')):
                        mesh = absolute_file_path
                    elif file_name.lower().endswith('.txt'):
                        description = absolute_file_path


                sql.insert_asset(conn, cursor, subfolder_name, image, mesh, description)
                logger.info(f"Inserted asset: {Fore.GREEN}{subfolder_name}{Fore.RESET}")


    def read(self):
        """Print out all the assets in the database."""
        # Get fresh connection and cursor for querying assets
        conn = self.db._get_connection()
        cursor = self.db._get_cursor()

        assets = sql.query_assets(cursor)
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
        """Return a list of all assets as dictionaries."""
        # Get fresh connection and cursor for querying assets
        conn = self.db._get_connection()
        cursor = self.db._get_cursor()

        assets = sql.query_assets(cursor)
        return [{
            "id": asset_id,
            "name": name,
            "image": image,
            "mesh": mesh,
            "description": description
        } for asset_id, name, image, mesh, description in assets]
