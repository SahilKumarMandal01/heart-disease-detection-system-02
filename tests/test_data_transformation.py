"""
Covers the following subcomponents ONLY:
    • read_data()
    • fix_outliers()
    • feature_engineering()
    • get_preprocessor()

Notes:
    - save_numpy_array_data, save_object, write_json_file are fully mocked.
    - Minimal strictness: tests verify shapes, presence of columns,
      and basic transformation behavior without enforcing exact numeric outputs.
"""

import json, os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

# Make project importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import src.components.data_transformation as data_transformation_module
from src.components.data_transformation import DataTransformation
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import TrainingPipelineConfig, DataTransformationConfig


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """
    Create minimal train/test CSV files matching required schema."""
    content = """age,sex,chestpaintype,restingbp,cholesterol,fastingbs,restingecg,maxhr,exerciseangina,oldpeak,st_slope,heartdisease
60,M,ATA,140,250,0,Normal,150,N,1.2,Up,1
45,F,NAP,130,230,0,ST,160,Y,0.3,Flat,0
50,M,ASY,120,300,1,LVH,170,N,2.5,Down,1
"""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    train_path.write_text(content)
    test_path.write_text(content)

    return str(train_path), str(test_path)


@pytest.fixture
def ingestion_artifact(sample_csv):
    train_path, test_path = sample_csv
    return DataIngestionArtifact(
        raw_file_path="unused",
        train_file_path=train_path,
        test_file_path=test_path,
        generated_schema="unused"
    )


@pytest.fixture
def validation_artifact():
    return DataValidationArtifact(
        report_file_path="unused",
        validation_status=True  # allow transformation to proceed
    )


@pytest.fixture
def transformation_config(tmp_path):
    tp = TrainingPipelineConfig()
    cfg = DataTransformationConfig(tp)

    # Override all file output paths (they will be mocked anyway)
    cfg.data_transformation_dir = str(tmp_path / "transform")
    cfg.transformer = str(tmp_path / "transform" / "transformer.pkl")
    cfg.training_data = str(tmp_path / "transform" / "train.npy")
    cfg.testing_data = str(tmp_path / "transform" / "test.npy")
    cfg.feature_names = str(tmp_path / "transform" / "feature_names.json")

    return cfg


@pytest.fixture
def dt(ingestion_artifact, validation_artifact, transformation_config, monkeypatch):
    """Create DataTransformation instance with save functions mocked."""
    monkeypatch.setattr(data_transformation_module, "save_numpy_array_data", MagicMock())
    monkeypatch.setattr(data_transformation_module, "save_object", MagicMock())
    monkeypatch.setattr(data_transformation_module, "write_json_file", MagicMock())

    return DataTransformation(
        data_ingestion_artifact=ingestion_artifact,
        data_validation_artifact=validation_artifact,
        data_transformation_config=transformation_config
    )


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

def test_read_data(dt, sample_csv):
    """read_data should load a CSV into a DataFrame with correct shape."""
    train_path, _ = sample_csv
    df = dt.read_data(train_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3
    assert "age" in df.columns
    assert "heartdisease" in df.columns


def test_fix_outliers(dt):
    """fix_outliers should clip extreme values downward (minimal strictness)."""
    df_train = pd.DataFrame({"x": [1, 2, 3, 100]})
    df_test = pd.DataFrame({"x": [2, 3, 200]})

    new_train, new_test = dt.fix_outliers(df_train.copy(), df_test.copy(), "x")

    # Minimal strictness: ensure clipping happened
    assert new_train["x"].max() < 100
    assert new_test["x"].max() < 200

def test_feature_engineering(dt, sample_csv):
    """Feature engineering should add bin labels and fill missing values minimally."""
    train_path, test_path = sample_csv
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    new_train, new_test = dt.feature_engineering(train_df.copy(), test_df.copy())

    # oldpeak becomes categorical after binning
    assert new_train["oldpeak"].dtype == "category"
    assert new_test["oldpeak"].dtype == "category"

    # restingbp should not contain NaN after imputation
    assert new_train["restingbp"].isna().sum() == 0
    assert new_test["restingbp"].isna().sum() == 0


def test_get_preprocessor(dt):
    """get_preprocessor should return a ColumnTransformer with expected named steps."""
    preprocessor, ordinal, nominal, numeric = dt.get_preprocessor()

    # Transformers list BEFORE fitting (transformers_ does not exist yet)
    names = [name for name, _, _ in preprocessor.transformers]

    assert "ordinal" in names
    assert "nominal" in names
    assert "numeric" in names

    # Minimal strictness for feature lists
    assert isinstance(ordinal, list)
    assert isinstance(nominal, list)
    assert isinstance(numeric, list)
    assert len(ordinal) > 0
    assert len(nominal) > 0
    assert len(numeric) > 0