"""
Scope:
- Test individual methods:
  * export_collection_as_dataframe
  * export_data_into_ingestion_store
  * save_data_schema
  * split_data_as_train_test

Notes:
- Uses mongomock (shared client) to avoid a real MongoDB.
- All file outputs are written inside pytest tmp_path.
- Overrides config paths and DB names on the DataIngestionConfig instance to keep tests hermetic.
"""

from pathlib import Path
from typing import List, Dict

import os, sys
import json
import pandas as pd
import numpy as np
import pytest
import mongomock

# Make project importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import the module under test 
import src.components.data_ingestion as data_ingestion_module
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from src.exception import HeartDiseaseException


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def env_vars(monkeypatch):
    """
    Ensure required environment variables are present for DataIngestion.__init__.
    DataIngestion checks MONGODB_URL at construction time.
    """
    monkeypatch.setenv("MONGODB_URL", "mongodb://dummy")
    yield


@pytest.fixture(autouse=True)
def shared_mongo_client(monkeypatch):
    """
    Provide a shared mongomock client for all tests so the code under test and
    the test assertions operate on the same in-memory database.
    """
    shared_client = mongomock.MongoClient()

    def _get_shared_client(*args, **kwargs):
        return shared_client

    monkeypatch.setattr(data_ingestion_module.pymongo, "MongoClient", _get_shared_client)
    yield shared_client


@pytest.fixture
def base_config(tmp_path):
    """
    Build a DataIngestionConfig backed by a TrainingPipelineConfig and
    override paths to use tmp_path for all file outputs.
    """
    tp = TrainingPipelineConfig()
    config = DataIngestionConfig(tp)

    # Force all file paths into tmp_path to avoid touching the repo
    artifact_dir = tmp_path / "artifacts"
    config.data_ingestion_dir = str(artifact_dir)
    config.raw_data_file_path = str(artifact_dir / "heart_disease.csv")
    config.train_data_file_path = str(artifact_dir / "train.csv")
    config.test_data_file_path = str(artifact_dir / "test.csv")
    config.schema_file_path = str(artifact_dir / "schema_generated.json")

    # Default split ratio; tests may override
    config.train_test_split_ratio = 0.2

    # DB names will be set by tests to stable values
    config.database_name = "test_db"
    config.collection_name = "test_collection"

    return config


# ---------------------------------------------------------------------
# Tests: export_collection_as_dataframe
# ---------------------------------------------------------------------

def test_export_collection_as_dataframe_success(env_vars, shared_mongo_client, base_config):
    """
    When the target collection has documents, export_collection_as_dataframe
    should return a pandas DataFrame without the MongoDB '_id' column.
    """
    # Arrange: insert sample documents into shared mongomock
    docs = [
        {"age": 63, "restingbp": 145, "cholesterol": 233, "oldpeak": 2.3, "heartdisease": 1},
        {"age": 45, "restingbp": 120, "cholesterol": 210, "oldpeak": 0.0, "heartdisease": 0},
        {"age": 55, "restingbp": 130, "cholesterol": 250, "oldpeak": 1.2, "heartdisease": 1},
    ]
    db = shared_mongo_client[base_config.database_name]
    coll = db[base_config.collection_name]
    coll.insert_many(docs)

    # Act
    ingestion = DataIngestion(base_config)
    df = ingestion.export_collection_as_dataframe()

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == len(docs)
    assert "_id" not in df.columns
    # Basic column presence
    for col in ("age", "restingbp", "cholesterol", "oldpeak", "heartdisease"):
        assert col in df.columns


def test_export_collection_as_dataframe_empty_raises(env_vars, shared_mongo_client, base_config):
    """
    If the collection is empty, export_collection_as_dataframe should raise HeartDiseaseException.
    """
    # Ensure collection is empty
    db = shared_mongo_client[base_config.database_name]
    coll = db[base_config.collection_name]
    coll.delete_many({})

    ingestion = DataIngestion(base_config)
    with pytest.raises(HeartDiseaseException):
        ingestion.export_collection_as_dataframe()


# ---------------------------------------------------------------------
# Tests: export_data_into_ingestion_store
# ---------------------------------------------------------------------

def test_export_data_into_ingestion_store_writes_csv(env_vars, base_config, tmp_path):
    """
    export_data_into_ingestion_store should persist the provided DataFrame to
    the configured raw_data_file_path and return the same DataFrame.
    """
    ingestion = DataIngestion(base_config)

    # Create a simple DataFrame
    df = pd.DataFrame({
        "age": [63, 45],
        "restingbp": [145, 120],
        "cholesterol": [233, 210],
        "oldpeak": [2.3, 0.0],
        "heartdisease": [1, 0],
    })

    result_df = ingestion.export_data_into_ingestion_store(df)
    assert result_df.equals(df)

    # File should exist on disk at raw_data_file_path
    raw_path = Path(base_config.raw_data_file_path)
    assert raw_path.exists()
    # Roundtrip: read the CSV and check contents
    persisted = pd.read_csv(raw_path)
    assert "age" in persisted.columns
    assert persisted.shape[0] == df.shape[0]


def test_export_data_into_ingestion_store_empty_raises(env_vars, base_config):
    """
    Trying to save an empty DataFrame should raise HeartDiseaseException.
    """
    ingestion = DataIngestion(base_config)
    empty_df = pd.DataFrame()
    with pytest.raises(HeartDiseaseException):
        ingestion.export_data_into_ingestion_store(empty_df)


# ---------------------------------------------------------------------
# Tests: save_data_schema
# ---------------------------------------------------------------------

def test_save_data_schema_creates_valid_json(env_vars, base_config, tmp_path):
    """
    save_data_schema should write a JSON file containing:
      - 'columns' with dtype strings
      - 'domain_range' for numeric columns present
      - 'unique_categorical_values' for categorical columns present
    """
    ingestion = DataIngestion(base_config)

    # Build DataFrame with numeric and categorical columns used in schema logic
    df = pd.DataFrame({
        "age": [30, 40, 50],
        "restingbp": [120, 130, 140],
        "cholesterol": [200, 220, 210],
        "maxhr": [150, 160, 170],
        "oldpeak": [0.0, 1.2, 0.5],
        "sex": ["M", "F", "M"],
        "chestpaintype": ["typical", "atypical", "non-anginal"],
        "restingecg": ["normal", "abnormal", "normal"],
        "exerciseangina": ["N", "Y", "N"],
        "st_slope": ["up", "flat", "down"],
        "fastingbs": [0, 1, 0],
        "heartdisease": [0, 1, 0],
    })

    # Ensure schema path is inside tmp_path
    base_config.schema_file_path = str(tmp_path / "schema_generated.json")

    ingestion.save_data_schema(df)

    schema_path = Path(base_config.schema_file_path).with_suffix(".json")
    assert schema_path.exists()

    content = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "schema" in content
    schema = content["schema"]
    assert "columns" in schema
    assert "domain_range" in schema
    assert "unique_categorical_values" in schema

    # Check domain range contains numeric keys we provided
    for key in ("age", "restingbp", "cholesterol", "maxhr", "oldpeak"):
        if key in df.columns:
            assert key in schema["domain_range"]
            assert isinstance(schema["domain_range"][key]["min"], float)
            assert isinstance(schema["domain_range"][key]["max"], float)

    # Check unique values for a sample categorical column
    assert "sex" in schema["unique_categorical_values"]
    assert set(schema["unique_categorical_values"]["sex"]) == {"M", "F"}


# ---------------------------------------------------------------------
# Tests: split_data_as_train_test
# ---------------------------------------------------------------------

def test_split_data_as_train_test_creates_files_and_preserves_stratify(env_vars, base_config, tmp_path):
    """
    split_data_as_train_test should write train/test CSVs to configured paths
    and preserve class distribution due to stratify parameter.
    """
    ingestion = DataIngestion(base_config)

    # Construct a DataFrame with balanced classes and enough rows for stratify
    n_per_class = 20
    df = pd.DataFrame({
        "age": list(range(n_per_class * 2)),
        "restingbp": np.random.randint(110, 160, size=2 * n_per_class),
        "cholesterol": np.random.randint(180, 280, size=2 * n_per_class),
        "oldpeak": np.random.random(size=2 * n_per_class),
        # binary target with equal counts
        "heartdisease": [0] * n_per_class + [1] * n_per_class,
    })

    # Override paths into tmp_path
    base_config.train_data_file_path = str(tmp_path / "train.csv")
    base_config.test_data_file_path = str(tmp_path / "test.csv")
    # Use a 30% test split for easier assertions
    base_config.train_test_split_ratio = 0.3

    train_path, test_path = ingestion.split_data_as_train_test(df)

    # Files exist
    assert Path(train_path).exists()
    assert Path(test_path).exists()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    total_rows = df.shape[0]
    expected_test_rows = int(total_rows * base_config.train_test_split_ratio)
    # Because of rounding, allow 1-row tolerance
    assert abs(test_df.shape[0] - expected_test_rows) <= 1
    assert train_df.shape[0] + test_df.shape[0] == total_rows

    # Check stratification roughly preserved: proportion of class 1 similar
    orig_prop = sum(df["heartdisease"]) / total_rows
    train_prop = sum(train_df["heartdisease"]) / train_df.shape[0]
    test_prop = sum(test_df["heartdisease"]) / test_df.shape[0]
    # Allow small tolerance due to discrete rounding
    assert abs(orig_prop - train_prop) < 0.15
    assert abs(orig_prop - test_prop) < 0.15


def test_split_data_as_train_test_not_enough_rows_raises(env_vars, base_config):
    """
    If the DataFrame has fewer than 2 rows, splitting should raise HeartDiseaseException.
    """
    ingestion = DataIngestion(base_config)
    df = pd.DataFrame({"age": [1], "heartdisease": [0]})
    with pytest.raises(HeartDiseaseException):
        ingestion.split_data_as_train_test(df)
