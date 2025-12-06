"""
Testing strategy:
- Use monkeypatch to set dummy environment variables.
- Mock kagglehub.dataset_download to return a temp directory with a synthetic CSV.
- Patch pymongo.MongoClient → mongomock.MongoClient so tests run fully in-memory.
- Test individual ETL methods (_validate_env, extract_data, transform_data, load_data).
"""

import os, sys
import pandas as pd
import numpy as np
import pytest
import mongomock

# -----------------------------------------
# Make project root importable
# -----------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import src.components.push_data as push_data_module
from src.components.push_data import HeartDiseaseETL
from src.exception import HeartDiseaseException


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture
def env_vars(monkeypatch):
    """Provide dummy env vars expected by HeartDiseaseETL."""
    monkeypatch.setenv("MONGODB_URL", "mongodb://dummy")
    monkeypatch.setenv("DATA_PATH", "dummy_kaggle_path")
    monkeypatch.setenv("DATABASE_NAME", "test_db")
    monkeypatch.setenv("COLLECTION_NAME", "test_collection")
    yield


@pytest.fixture
def temp_csv_dir(tmp_path):
    """
    Create a temporary directory containing a CSV file mimicking the Kaggle dataset.
    """
    df = pd.DataFrame({
        "Age ": [63, 67, 67],
        "RestingBP": [145, 0, 120],
        "Cholesterol": [233, 0, 300],
        "OldPeak": [-2.3, 1.2, -1.1],
        "Target": [1, 1, 0]
    })

    dirpath = tmp_path / "kaggle_download"
    dirpath.mkdir()
    csv_path = dirpath / "heart.csv"
    df.to_csv(csv_path, index=False)
    return str(dirpath)


@pytest.fixture(autouse=True)
def patch_mongo_client(monkeypatch):
    """
    Patch MongoClient so all instances share the same in-memory mongomock server.
    This ensures the ETL write and the test read use the same DB state.
    """
    shared_client = mongomock.MongoClient()

    def _shared_mongo_client(*args, **kwargs):
        return shared_client

    monkeypatch.setattr(push_data_module.pymongo, "MongoClient", _shared_mongo_client)
    yield


# --------------------------------------------------------------------------------------
# Tests for environment validation
# --------------------------------------------------------------------------------------

def test_validate_env_missing(monkeypatch):
    """If env vars are missing, constructor should raise HeartDiseaseException."""
    for key in ("MONGODB_URL", "DATA_PATH", "DATABASE_NAME", "COLLECTION_NAME"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(HeartDiseaseException):
        HeartDiseaseETL()


def test_validate_env_present(env_vars):
    """With all variables present, constructor should initialize correctly."""
    etl = HeartDiseaseETL()
    assert etl.mongodb_url == "mongodb://dummy"
    assert etl.dataset_path == "dummy_kaggle_path"
    assert etl.database == "test_db"
    assert etl.collection == "test_collection"


# --------------------------------------------------------------------------------------
# Tests for extract_data()
# --------------------------------------------------------------------------------------

def test_extract_data_no_csv(env_vars, monkeypatch, tmp_path):
    """If download dir has no CSV files, extract_data should raise HeartDiseaseException."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    monkeypatch.setattr(push_data_module.kagglehub, "dataset_download", lambda path: str(empty_dir))

    etl = HeartDiseaseETL()
    with pytest.raises(HeartDiseaseException):
        etl.extract_data()


def test_extract_data_success(env_vars, temp_csv_dir, monkeypatch):
    """extract_data should return a non-empty DataFrame."""
    monkeypatch.setattr(push_data_module.kagglehub, "dataset_download", lambda path: temp_csv_dir)

    etl = HeartDiseaseETL()
    df = etl.extract_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Age " in df.columns or any("age" in c.lower() for c in df.columns)


# --------------------------------------------------------------------------------------
# Tests for transform_data()
# --------------------------------------------------------------------------------------

def test_transform_data_basic_transform(env_vars):
    """
    Verify transform_data:
    - Normalizes column names
    - Replaces zero values with NaN
    - Converts negative oldpeak to absolute
    - Drops exact duplicates only
    """
    etl = HeartDiseaseETL()

    raw_df = pd.DataFrame({
        "Age ": [63, 63],
        "RestingBP": [120, 0],
        "Cholesterol": [245, 0],
        "OldPeak": [-1.5, -1.5],
        "Some-Col!": [1, 1]
    })

    records = etl.transform_data(raw_df)
    assert isinstance(records, list)

    # These rows are NOT exact duplicates → expect 2 records
    assert len(records) == 2

    df = pd.DataFrame(records)

    # Column normalization
    expected_cols = {"age", "restingbp", "cholesterol", "oldpeak", "somecol"}
    assert expected_cols.issubset(set(df.columns))

    # Zero → NaN
    assert df.loc[1, "restingbp"] is None or np.isnan(df.loc[1, "restingbp"])
    assert df.loc[1, "cholesterol"] is None or np.isnan(df.loc[1, "cholesterol"])

    # Negative oldpeak → absolute
    assert df["oldpeak"].min() >= 0


def test_transform_data_empty_dataframe(env_vars):
    """Empty DataFrame should raise HeartDiseaseException."""
    etl = HeartDiseaseETL()
    with pytest.raises(HeartDiseaseException):
        etl.transform_data(pd.DataFrame())


# --------------------------------------------------------------------------------------
# Tests for load_data()
# --------------------------------------------------------------------------------------

def test_load_data_no_records(env_vars):
    """load_data should return 0 when no records are provided."""
    etl = HeartDiseaseETL()
    assert etl.load_data([]) == 0


def test_load_data_inserts_records(env_vars):
    """
    load_data should insert records into mongomock and return inserted count.
    """
    etl = HeartDiseaseETL()

    records = [
        {"age": 63, "restingbp": 120, "cholesterol": 233, "oldpeak": 1.2, "target": 1},
        {"age": 45, "restingbp": 130, "cholesterol": 210, "oldpeak": 0.0, "target": 0},
    ]

    inserted = etl.load_data(records)
    assert inserted == 2

    # Validate DB contents
    client = push_data_module.pymongo.MongoClient(etl.mongodb_url)
    db = client[etl.database]
    coll = db[etl.collection]
    assert coll.count_documents({}) == 2
