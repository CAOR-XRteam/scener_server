import os
import sqlite3
import json
from typing import Optional

from beartype import beartype
from colorama import Fore
from library.sql.row import SQL
from library.manager.database import Database as DB
from loguru import logger
from pydantic import BaseModel, Field

from agent.llm.creation import initialize_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# TODO: more precise error handling to propagate to the agent


class AppAsset(BaseModel):
    id: str
    name: str
    image: str
    mesh: str
    description: str


class NullableAppAsset(BaseModel):
    asset: Optional[AppAsset] = Field(None)


@beartype
class Library:
    def __init__(self, db: DB):
        self.db = db

    def fill(self, path: str):
        """Fill the database with assets from the specified directory."""
        try:
            cursor = self.db._get_cursor()  # fresh cursor
        except Exception as e:
            logger.error(f"Failed to get a connection or cursor: {e}")
            raise

        if not os.path.exists(path):
            logger.error(f"Path to fill from does not exists: {path}")
            raise FileNotFoundError(f"Path to fill from does not exists: {path}")
        if not os.path.isdir(path):
            logger.error(f"Path to fill from is not a directory: {path}")
            raise NotADirectoryError(f"Path to fill from is not a directory: {path}")

        try:
            subfolder_names = os.listdir(path)
        except OSError as e:
            logger.error(f"Failed to list directory {path}: {e}")
            raise

        for subfolder_name in subfolder_names:
            subpath = os.path.join(path, subfolder_name)
            if os.path.isdir(subpath):
                image = mesh = description = None
                try:
                    for file_name in os.listdir(subpath):
                        file_path = os.path.join(subpath, file_name)
                        absolute_file_path = os.path.abspath(file_path)

                        if file_name.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".webp")
                        ):
                            image = absolute_file_path
                        elif file_name.lower().endswith(
                            (".obj", ".fbx", ".stl", ".ply", ".glb")
                        ):
                            mesh = absolute_file_path
                        elif file_name.lower().endswith(".txt"):
                            description = absolute_file_path
                    SQL.insert_asset(
                        self.db._conn, cursor, subfolder_name, image, mesh, description
                    )
                    logger.info(
                        f"Inserted asset: {Fore.GREEN}{subfolder_name}{Fore.RESET}"
                    )
                except OSError as e:
                    logger.error(f"Failed to list subdirectory {subpath}: {e}")
                except sqlite3.Error as e:
                    logger.error(f"Failed to insert asset {subfolder_name}: {e}")

    def read(self):
        """Print out all the assets in the database."""
        # Get fresh connection and cursor for querying assets
        try:
            cursor = self.db._get_cursor()
            assets = SQL.query_assets(cursor)
            if assets:
                print(
                    f"{'ID':<4} {'Name':<10} {'Image':<10} {'Mesh':<10} {'Description':<10}"
                )
                for asset in assets:
                    asset_id, asset_name, asset_image, asset_mesh, asset_description = (
                        asset
                    )
                    name = f"{Fore.YELLOW}{asset_name:<10}{Fore.RESET}"
                    img = (
                        f"{Fore.GREEN}{'ok':<10}{Fore.RESET}"
                        if asset_image
                        else f"{Fore.RED}{'None':<10}{Fore.RESET}"
                    )
                    mesh = (
                        f"{Fore.GREEN}{'ok':<10}{Fore.RESET}"
                        if asset_mesh
                        else f"{Fore.RED}{'None':<10}{Fore.RESET}"
                    )
                    desc = (
                        f"{Fore.GREEN}{'ok':<10}{Fore.RESET}"
                        if asset_description
                        else f"{Fore.RED}{'None':<10}{Fore.RESET}"
                    )
                    print(f"{asset_id:<4} {name} {img} {mesh} {desc}")
            else:
                print("No assets found.")
        except Exception as e:
            logger.error(f"Failed to read assets from the database: {e}")
            raise

    def get_list(self):
        """Return a list of all assets as dictionaries."""
        # Get fresh connection and cursor for querying assets
        try:
            cursor = self.db._get_cursor()
            assets = SQL.query_assets(cursor)
            return [
                AppAsset(
                    id=str(asset_id),
                    name=name,
                    image=image,
                    mesh=mesh,
                    description=description,
                )
                for asset_id, name, image, mesh, description in assets
            ]
        except Exception as e:
            logger.error(f"Failed to read assets from the database: {e}")
            raise

    def get_asset(self, name: str):
        """Return asset by its name"""
        try:
            cursor = self.db._get_cursor()
            asset = SQL.query_asset_by_name(cursor, name)

            if asset:
                return AppAsset(
                    id=str(asset[0]),
                    name=asset[1],
                    image=asset[2],
                    mesh=asset[3],
                    description=asset[4],
                )
            else:
                raise ValueError(f"Asset {name} not found")
        except Exception as e:
            logger.error(f"Failed to get asset from the database: {e}")
            raise


# pip install langchain-chroma langchain sentence-transformers
@beartype
class AssetFinder:
    def __init__(self, assets: list[AppAsset]):
        self.threshold = 0.8
        self.asset_map = {asset.id: asset for asset in assets}

        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.vector_store = Chroma(
            collection_name="app_assets",
            embedding_function=embedding_function,
            persist_directory="./asset_db",
        )

        self._populate_db(assets)

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        self.llm = initialize_model("gemma3:12b")
        self.rerank_chain = self._create_rerank_chain()

    def _populate_db(self, assets: list[AppAsset]):
        existing_ids = self.vector_store.get(include=[])["ids"]
        new_assets = [asset for asset in assets if asset.id not in existing_ids]

        if not new_assets:
            logger.info("ChromaDB collection is already up-to-date.")
            return

        logger.info(f"Adding {len(new_assets)} new assets to ChromaDB.")

        new_documents = [
            Document(
                page_content=asset.description,
                metadata={"id": asset.id, "name": asset.name},
            )
            for asset in new_assets
        ]

        self.vector_store.add_documents(
            new_documents, ids=[asset.id for asset in new_assets]
        )

    def _create_rerank_chain(self):
        few_shot_examples = """
    ---
    EXAMPLE 1: A "close but not exact" match, which MUST be rejected.

    [USER]
    Target Description:
    "a persian cat with white and orange spots"

    Candidate Assets:
    [
        {"id": "cat-01", "name": "Siamese Cat", "description": "A siamese cat with striking blue eyes and cream-colored fur."},
        {"id": "cat-02", "name": "White Cat", "description": "A fluffy white cat with green eyes."}
    ]

    Instructions:
    - Analyze the target description for specific, mandatory details.
    - Scrutinize the candidates and select ONLY the one that meets every single detail.
    - If no candidate is a perfect match, you MUST return null.

    Respond with ONLY the required JSON object.
    [END USER]

    [ASSISTANT]
    {"asset": null}
    [END ASSISTANT]

    ---
    EXAMPLE 2: A perfect match, which should be selected.

    [USER]
    Target Description:
    "a common brown tabby cat with green eyes"

    Candidate Assets:
    [
        {"id": "cat-02", "name": "Tabby Cat", "description": "A common brown tabby cat with green eyes."},
        {"id": "dog-01", "name": "Golden Retriever", "description": "A friendly golden retriever dog."}
    ]

    Instructions:
    - Analyze the target description for specific, mandatory details.
    - Scrutinize the candidates and select ONLY the one that meets every single detail.
    - If no candidate is a perfect match, you MUST return null.

    Respond with ONLY the required JSON object.
    [END USER]

    [ASSISTANT]
    {"asset": {"id": "cat-02", "name": "Tabby Cat", "description": "A common brown tabby cat with green eyes.", "image": null, "mesh": null}}
    [END ASSISTANT]
    ---
    """

        system_prompt = f"""
        You are a hyper-critical and pedantic validation engine. Your ONLY job is to determine if any of the given candidate assets is an EXACT match for a target description. You MUST be extremely strict.

        **PRIMARY DIRECTIVE: ZERO TOLERANCE FOR MISMATCHES.**

        You will be given a 'Target Description' and a list of 'Candidate Assets'. You must follow this logic precisely:
        
        1.  **Deconstruct Requirements:** Break down the 'Target Description' into a checklist of non-negotiable attributes. For "a persian cat with white and orange spots", the checklist is [subject: 'cat', breed: 'persian', color: 'white', color: 'orange spots'].

        2.  **Verify ALL Checklist Items:** For each candidate, you must verify that its description satisfies EVERY SINGLE item on the checklist.
            - If a candidate's description is "a fluffy white cat", it fails the checklist because 'persian' and 'orange spots' are missing.
            - If ANY item from the checklist is not explicitly met by the candidate's description, that candidate is an IMMEDIATE failure.

        3.  **Return Decision:**
            - If one candidate passes the 100% verification check, return its full JSON object.
            - **If NO candidate satisfies ALL checklist items, you MUST return null.** This is the most common and expected outcome. Do not "settle" for the closest match.

        Your response MUST be a JSON object with a single key "asset", which is either the full asset JSON or `null`.

        Study the following examples to understand the required level of strictness:
        {few_shot_examples}
        """

        user_prompt = """
            Target Description:
            {description}

            Available Assets:
            {assets}

            Instructions:
            - Compare the target description with the descriptions of all assets.
            - Return the single best matching asset.
            - If no asset matches closely enough, return null.
            - Be precise and conservative. Do not guess.
            - Apply the zero-tolerance validation logic.
            - Return the JSON for a perfect match or null if none exists.

            You must respond ONLY with the JSON object of the best matching asset, or null if no match is found. Do not include any other text, explanations, or code.
            """
        parser = JsonOutputParser(pydantic_object=NullableAppAsset)
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt)]
        )
        return prompt | self.llm | parser

    @beartype
    def find_by_description(self, description: str) -> AppAsset | None:
        try:
            logger.info(f"Starting asset search for: '{description}'")

            candidate_docs = self.vector_store.similarity_search_with_relevance_scores(
                description, k=10
            )

            if not candidate_docs:
                logger.info("Semantic search returned no results.")
                return None

            strong_candidates_docs = [
                doc for doc, score in candidate_docs if score >= self.threshold
            ]

            if not strong_candidates_docs:
                logger.info(f"No candidates met the threshold of {self.threshold}.")
                return None

            candidates = [
                self.asset_map[doc.metadata["id"]]
                for doc in strong_candidates_docs
                if doc.metadata["id"] in self.asset_map
            ]
            candidates_json = json.dumps([asset.model_dump() for asset in candidates])

            result: NullableAppAsset = self.rerank_chain.invoke(
                {"description": description, "assets": candidates_json}
            )

            if result and result.asset:
                logger.info(f"LLM re-ranking selected asset ID: {result.asset.id}")
                return result.asset
            else:
                logger.info(
                    "LLM re-ranking concluded no asset was a sufficiently close match."
                )
                return None

        except Exception as e:
            logger.error(f"Error while searching for an asset: {e}")
            return None
