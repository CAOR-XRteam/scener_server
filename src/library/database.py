from library import sql
from loguru import logger
from colorama import Fore, Style
import json
import os


# Database management
class DB:
    def __init__(self, path):
        self.path = path
        self.conn = sql.connect_db(self.path)
        self.cursor = sql.get_cursor(self.conn)
        self.init()

    def init(self):
        sql.create_table_asset(self.conn, self.cursor)
        logger.success(f"Connected to database {Fore.GREEN}{self.path}{Fore.GREEN}{Fore.RESET}")

    def fill(self, path):
        """Recursively explore a folder and fill the database with asset data."""
        for subfolder_name in os.listdir(path):
            subpath = os.path.join(path, subfolder_name)

            # Only process subfolders (ignore files)
            if os.path.isdir(subpath):
                # Find files for image, mesh, and description
                image_file = None
                mesh_file = None
                description_file = None

                for file_name in os.listdir(subpath):
                    file_path = os.path.join(subpath, file_name)

                    # Match image files by extension
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_file = file_path
                    # Match mesh files by extension
                    elif file_name.lower().endswith(('.obj', '.fbx', '.stl', '.ply', '.glb')):
                        mesh_file = file_path
                    # Match description file (e.g., txt)
                    elif file_name.lower().endswith('.txt'):
                        description_file = file_path

                # Assign default values if any of the expected files are missing
                image = image_file if image_file else None
                mesh = mesh_file if mesh_file else None
                description = description_file if description_file else None

                # Insert the asset into the database
                sql.insert_asset(self.conn, self.cursor, subfolder_name, image, mesh, description)
                logger.info(f"Inserted asset: {Fore.GREEN}{subfolder_name}{Fore.RESET}")

    def read(self):
        """Fetch and display all assets from the database in a human-readable format."""
        # Fetch all assets
        assets = sql.query_assets(self.cursor)

        # Print the results in a nice format
        if assets:
            print(f"{'ID':<4} {'Name':<10} {'Image':<10} {'Mesh':<10} {'Description':<10}")
            for asset in assets:
                asset_id, asset_name, asset_image, asset_mesh, asset_description = asset
                id = f"{asset_id:<4}"
                name = f"{Fore.YELLOW}{asset_name:<10}{Fore.RESET}"
                img = f"{Fore.GREEN}{"ok":<10}{Fore.RESET}" if asset_image else f"{Fore.RED}{"None":<10}{Fore.RESET}"
                mesh = f"{Fore.GREEN}{"ok":<10}{Fore.RESET}" if asset_mesh else f"{Fore.RED}{"None":<10}{Fore.RESET}"
                description = f"{Fore.GREEN}{"ok":<10}{Fore.RESET}" if asset_description else f"{Fore.RED}{"None":<10}{Fore.RESET}"
                print(f"{id} {name} {img} {mesh} {description}")
        else:
            print("No assets found.")

    def get_list(self):
        """Return the content of the asset table as a list of dictionaries."""
        assets = sql.query_assets(self.cursor)
        asset_list = []

        for asset in assets:
            asset_id, name, image, mesh, description = asset
            asset_list.append({
                "id": asset_id,
                "name": name,
                "image": image,
                "mesh": mesh,
                "description": description
            })

        return asset_list

    def close(self):
        sql.close_connection(self.conn)
