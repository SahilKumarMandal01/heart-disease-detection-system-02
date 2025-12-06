import os
import sys
import json
from pathlib import Path
from typing import List, Dict

import certifi
import pandas as pd
import numpy as np
import pymongo
import kagglehub

from dotenv import load_dotenv
from src.exception import HeartDiseaseException
from src.logging import logging

load_dotenv()


class HeartDiseaseETL:
    """
    ETL pipeline for the Heart Disease dataset:
      1. Extract: download dataset from Kaggle
      2. Transform: clean + preprocess
      3. Load: store processed records in MongoDB
    """

    def __init__(self) -> None:
        """Validate environment variables and initialize config."""
        try:
            self.mongodb_url: str = os.getenv("MONGODB_URL")
            self.dataset_path: str = os.getenv("DATA_PATH")
            self.database: str = os.getenv("DATABASE_NAME")
            self.collection: str = os.getenv("COLLECTION_NAME")
            self.ca_file: str = certifi.where()

            self._validate_env()

            logging.info("💡 HeartDiseaseETL initialized successfully.")

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ------------------------------------------------------------------ #

    def _validate_env(self) -> None:
        """Validate required environment variables."""
        required = {
            "MONGODB_URL": self.mongodb_url,
            "DATA_PATH": self.dataset_path,
            "DATABASE_NAME": self.database,
            "COLLECTION_NAME": self.collection,
        }

        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    # ------------------------------------------------------------------ #

    def extract_data(self) -> pd.DataFrame:
        """Download dataset from Kaggle and return as DataFrame."""
        try:
            logging.info("📥 Extracting Heart Disease dataset from Kaggle...")

            dataset_dir = kagglehub.dataset_download(self.dataset_path)
            dataset_dir = Path(dataset_dir)

            csv_files = list(dataset_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the downloaded dataset directory.")

            csv_path = csv_files[0]
            df = pd.read_csv(csv_path)

            if df.empty:
                raise ValueError(f"Dataset at {csv_path} is empty.")

            logging.info(
                "📄 Dataset loaded",
                extra={"file": str(csv_path), "shape": df.shape}
            )

            return df

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ------------------------------------------------------------------ #

    def transform_data(self, df: pd.DataFrame) -> List[Dict]:
        """Clean and transform DataFrame into list-of-dicts."""
        try:
            logging.info("🧹 Transforming dataset...")

            if df.empty:
                raise ValueError("Cannot transform empty DataFrame.")

            # Normalize column names
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
                .str.replace(r"[^\w_]", "", regex=True)
            )

            # Replace impossible values
            replace_zero_columns = ["restingbp", "cholesterol"]
            for col in replace_zero_columns:
                if col in df:
                    zero_count = (df[col] == 0).sum()
                    if zero_count:
                        logging.warning(
                            f"{zero_count} zero values found in '{col}'. Replacing with NaN."
                        )
                    df[col] = df[col].replace(0, np.nan)

            # Fix negative oldpeak
            if "oldpeak" in df:
                negative_count = (df["oldpeak"] < 0).sum()
                if negative_count > 0:
                    logging.warning(
                        f"{negative_count} negative 'oldpeak' values found. Converting to absolute."
                    )
                df["oldpeak"] = df["oldpeak"].abs()

            # Deduplicate
            df = df.drop_duplicates().reset_index(drop=True)

            # Convert to dicts for MongoDB
            records = df.to_dict(orient="records")

            logging.info(
                "✨ Transformation complete.",
                extra={"records": len(records)}
            )

            return records

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ------------------------------------------------------------------ #

    def load_data(self, records: List[Dict]) -> int:
        """Insert transformed records into MongoDB."""
        try:
            logging.info("🗄️ Connecting to MongoDB...")

            if not records:
                logging.warning("No records to load. Skipping insert.")
                return 0

            client = pymongo.MongoClient(self.mongodb_url, tlsCAFile=self.ca_file)
            db = client[self.database]
            collection = db[self.collection]

            # Clear collection before insert
            collection.delete_many({})

            result = collection.insert_many(records)
            inserted_count = len(result.inserted_ids)

            logging.info(
                "📌 MongoDB load successful",
                extra={"inserted_records": inserted_count}
            )

            return inserted_count

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ------------------------------------------------------------------ #

    def run_etl(self) -> None:
        """Run the whole ETL pipeline."""
        try:
            logging.info("🚀 Starting ETL pipeline...")
            df = self.extract_data()
            records = self.transform_data(df)
            inserted = self.load_data(records)

            logging.info(f"✅ ETL complete. Total inserted: {inserted}")
            print(f"\nETL Completed Successfully 🎉 — {inserted} records inserted.\n")

        except Exception as e:
            logging.error(f"❌ ETL pipeline failed: {e}")
            raise HeartDiseaseException(e, sys)


# ------------------------------------------------------------------ #

if __name__ == "__main__":
    try:
        etl = HeartDiseaseETL()
        etl.run_etl()
    except Exception as e:
        logging.error(f"ETL execution failed: {e}")
        sys.exit(1)
