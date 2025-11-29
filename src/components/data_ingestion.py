import os
import sys
import certifi
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import pymongo

from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src.exception import HeartDiseaseException
from src.logging import logging
from src.utils.main_utils.utils import write_json_file
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

# Load environment variables
load_dotenv()


class DataIngestion:
    """
    Data ingestion pipeline for Heart Disease ML workflow.

    Steps:
      1. Read processed records from MongoDB.
      2. Save raw dataset to ingestion storage.
      3. Generate data schema as JSON.
      4. Split train/test datasets.
      5. Produce DataIngestionArtifact.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        try:
            if not isinstance(data_ingestion_config, DataIngestionConfig):
                raise ValueError("data_ingestion_config must be a DataIngestionConfig instance.")

            self.config = data_ingestion_config
            self.ca_file = certifi.where()

            self.mongodb_url = os.getenv("MONGODB_URL")
            if not self.mongodb_url:
                raise ValueError("Missing required environment variable: MONGODB_URL")

            logging.info("🚀 DataIngestion initialized successfully.")

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ------------------------------------------------------------------ #
    # Helper — Get Mongo Collection
    # ------------------------------------------------------------------ #

    def _get_collection(self) -> Tuple[pymongo.MongoClient, pymongo.collection.Collection]:
        """Return (client, collection)."""
        try:
            client = pymongo.MongoClient(self.mongodb_url, tlsCAFile=self.ca_file)
            db = client[self.config.database_name]
            collection = db[self.config.collection_name]
            return client, collection
        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ------------------------------------------------------------------ #
    # Step 1 — Fetch Data
    # ------------------------------------------------------------------ #

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """Fetch the MongoDB collection as DataFrame."""
        try:
            logging.info("📥 Fetching dataset from MongoDB...")

            client, collection = self._get_collection()

            try:
                records = list(collection.find())
            finally:
                client.close()

            if not records:
                raise HeartDiseaseException(
                    f"No documents found in {self.config.database_name}.{self.config.collection_name}",
                    sys
                )

            df = pd.DataFrame(records)

            # Remove MongoDB internal IDs
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            logging.info(
                "📄 DataFrame loaded",
                extra={"rows": df.shape[0], "columns": df.shape[1]}
            )

            return df

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ------------------------------------------------------------------ #
    # Step 2 — Save Raw Dataset
    # ------------------------------------------------------------------ #

    def export_data_into_ingestion_store(self, df: pd.DataFrame) -> pd.DataFrame:
        """Save raw dataset CSV to ingestion directory."""
        try:
            if df.empty:
                raise HeartDiseaseException("Cannot save empty DataFrame.", sys)

            raw_path = Path(self.config.raw_data_file_path)
            raw_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(raw_path, index=False)
            logging.info(f"💾 Raw dataset saved at: {raw_path}")

            return df

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ------------------------------------------------------------------ #
    # Step 3 — Generate Schema JSON
    # ------------------------------------------------------------------ #

    def save_data_schema(self, df: pd.DataFrame) -> None:
        """Generate schema JSON including column types, domain ranges, unique values."""
        try:
            if df.empty:
                raise HeartDiseaseException("Cannot generate schema from empty DataFrame.", sys)

            schema_path = Path(self.config.schema_file_path).with_suffix(".json")
            schema_path.parent.mkdir(parents=True, exist_ok=True)

            # Column + dtype
            columns = {col: str(dtype) for col, dtype in df.dtypes.items()}

            # Numeric domain ranges
            numeric_cols = ["age", "restingbp", "cholesterol", "maxhr", "oldpeak"]
            domain_range = {
                col: {"min": float(df[col].min()), "max": float(df[col].max())}
                for col in numeric_cols if col in df.columns
            }

            # Categorical unique values
            categorical_cols = [
                "sex", "chestpaintype", "restingecg",
                "exerciseangina", "st_slope", "fastingbs", "heartdisease"
            ]

            unique_values = {
                col: df[col].dropna().unique().tolist()
                for col in categorical_cols if col in df.columns
            }

            schema_content = {
                "schema": {
                    "columns": columns,
                    "domain_range": domain_range,
                    "unique_categorical_values": unique_values
                }
            }

            write_json_file(str(schema_path), schema_content)
            logging.info(f"📐 Schema JSON saved at: {schema_path}")

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ------------------------------------------------------------------ #
    # Step 4 — Train/Test Split
    # ------------------------------------------------------------------ #

    def split_data_as_train_test(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Split dataset and save train/test CSVs."""
        try:
            if df.shape[0] < 2:
                raise HeartDiseaseException(
                    f"Not enough samples to split (rows={df.shape[0]})", sys
                )

            logging.info("✂️ Splitting dataset...", extra={"rows": df.shape[0]})

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.train_test_split_ratio,
                random_state=42,
                stratify=df["heartdisease"]
            )

            train_path = Path(self.config.train_data_file_path)
            test_path = Path(self.config.test_data_file_path)

            train_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.parent.mkdir(parents=True, exist_ok=True)

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logging.info(f"🟩 Train saved: {train_path} | shape={train_df.shape}")
            logging.info(f"🟦 Test saved:  {test_path} | shape={test_df.shape}")

            return str(train_path), str(test_path)

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ------------------------------------------------------------------ #
    # Step 5 — Full Pipeline
    # ------------------------------------------------------------------ #

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Execute complete ingestion pipeline."""
        try:
            logging.info("🚀 Starting Data Ingestion pipeline...")

            df = self.export_collection_as_dataframe()
            df = self.export_data_into_ingestion_store(df)
            self.save_data_schema(df)
            train_path, test_path = self.split_data_as_train_test(df)

            artifact = DataIngestionArtifact(
                raw_file_path=self.config.raw_data_file_path,
                train_file_path=train_path,
                test_file_path=test_path,
                generated_schema=str(Path(self.config.schema_file_path).with_suffix(".json"))
            )

            logging.info("🎉 Data ingestion completed successfully!", extra={"artifact": artifact.__dict__})
            return artifact

        except Exception as e:
            raise HeartDiseaseException(e, sys)
