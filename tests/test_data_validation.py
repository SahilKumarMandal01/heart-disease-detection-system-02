"""
Covers:
- validate_schema
- validate_domain_ranges
- validate_categorical_values
- generate_report (writes JSON report; returns boolean)

Approach:
- Create temporary predefined_schema and generated_schema JSON files under pytest tmp_path.
- Construct DataIngestionArtifact pointing to the generated schema file.
- Construct DataValidationConfig and override predefined_schema path and report_file_path.
- Use real read_json_file / write_json_file behavior (they are lightweight and deterministic).
"""

import os, sys
import json
from pathlib import Path
import pytest

# Make project importable when running tests directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.components.data_validation import DataValidation
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import TrainingPipelineConfig, DataValidationConfig
from src.exception import HeartDiseaseException


# -------------------------
# Helper fixtures
# -------------------------

@pytest.fixture
def predefined_schema_file(tmp_path):
    """
    Write the provided predefined_schema JSON into a temporary file and return its path.
    This schema matches the format expected by DataValidation (_predefined_schema_ in your project).
    """
    schema = {
        "version": "1.0",
        "schema": {
            "columns": {
                "age": "int64",
                "sex": "object",
                "chestpaintype": "object",
                "restingbp": "float64",
                "cholesterol": "float64",
                "fastingbs": "int64",
                "restingecg": "object",
                "maxhr": "int64",
                "exerciseangina": "object",
                "oldpeak": "float64",
                "st_slope": "object",
                "heartdisease": "int64"
            },
            "domain_constraints": {
                "age": {"min": 0, "max": 120},
                "restingbp": {"min": 1, "max": 250},
                "cholesterol": {"min": 1, "max": 700},
                "maxhr": {"min": 40, "max": 250},
                "oldpeak": {"min": 0, "max": 10}
            },
            "categorical_allowed": {
                "sex": ["M", "F"],
                "chestpaintype": ["ATA", "NAP", "ASY", "TA"],
                "restingecg": ["Normal", "ST", "LVH"],
                "exerciseangina": ["Y", "N"],
                "st_slope": ["Up", "Flat", "Down"],
                "fastingbs": [0, 1],
                "heartdisease": [0, 1]
            }
        }
    }

    p = tmp_path / "schema_predefined.json"
    p.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return str(p)


@pytest.fixture
def generated_schema_file(tmp_path):
    """
    Create a valid generated schema JSON (the artifact that would be produced by data ingestion).
    Contains columns, domain_range and unique_categorical_values sections.
    """
    gen = {
        "schema": {
            "columns": {
                "age": "int64",
                "sex": "object",
                "chestpaintype": "object",
                "restingbp": "float64",
                "cholesterol": "float64",
                "fastingbs": "int64",
                "restingecg": "object",
                "maxhr": "int64",
                "exerciseangina": "object",
                "oldpeak": "float64",
                "st_slope": "object",
                "heartdisease": "int64"
            },
            "domain_range": {
                "age": {"min": 25.0, "max": 80.0},
                "restingbp": {"min": 90.0, "max": 170.0},
                "cholesterol": {"min": 150.0, "max": 330.0},
                "maxhr": {"min": 100.0, "max": 190.0},
                "oldpeak": {"min": 0.0, "max": 6.0}
            },
            "unique_categorical_values": {
                "sex": ["M", "F"],
                "chestpaintype": ["ATA", "NAP"],
                "restingecg": ["Normal", "ST"],
                "exerciseangina": ["Y", "N"],
                "st_slope": ["Up", "Flat"],
                "fastingbs": [0, 1],
                "heartdisease": [0, 1]
            }
        }
    }

    p = tmp_path / "schema_generated.json"
    p.write_text(json.dumps(gen, indent=2), encoding="utf-8")
    return str(p)


@pytest.fixture
def data_validation_config(tmp_path, predefined_schema_file):
    """
    Construct a DataValidationConfig and override the predefined_schema and report file paths
    to use the pytest tmp_path.
    """
    tp = TrainingPipelineConfig()
    cfg = DataValidationConfig(tp)

    # override to tmp_path locations to avoid writing into repo
    cfg.predefined_schema = str(predefined_schema_file)
    cfg.data_validation_dir = str(tmp_path / "validation")
    cfg.report_file_path = str(Path(cfg.data_validation_dir) / "validation_report.json")
    return cfg


@pytest.fixture
def ingestion_artifact(generated_schema_file):
    """
    Create a DataIngestionArtifact that points to the generated schema file produced in tests.
    Raw/train/test file paths are not used by DataValidation internals, but the artifact requires them.
    """
    return DataIngestionArtifact(
        raw_file_path="unused",
        train_file_path="unused",
        test_file_path="unused",
        generated_schema=str(generated_schema_file)
    )


# -------------------------
# Tests
# -------------------------

def test_init_with_invalid_types_raises(predefined_schema_file, generated_schema_file):
    """Constructor should raise if given wrong types for artifact or config."""
    tp = TrainingPipelineConfig()
    cfg = DataValidationConfig(tp)

    # Wrong type for artifact
    with pytest.raises(HeartDiseaseException):
        DataValidation(data_ingestion_artifact="not_an_artifact", data_validation_config=cfg)

    # Wrong type for config
    dummy_artifact = DataIngestionArtifact("a", "b", "c", generated_schema_file)
    with pytest.raises(HeartDiseaseException):
        DataValidation(data_ingestion_artifact=dummy_artifact, data_validation_config="not_a_config")


def test_validate_schema_pass_and_fail(data_validation_config, ingestion_artifact, tmp_path):
    """
    validate_schema should return PASSED for matching columns/dtypes and FAILED
    when there are missing/unexpected/dtype mismatches.
    """
    dv = DataValidation(ingestion_artifact, data_validation_config)

    # Load the schemas
    generated_schema = dv.generated_schema
    predefined_schema = dv.predefined_schema

    # Sanity: matching schema -> PASS
    res = dv.validate_schema(generated_schema, predefined_schema)
    assert res["status"] == "PASSED"
    assert res["missing_columns"] == []
    assert res["unexpected_columns"] == []
    assert res["dtype_mismatches"] == {}

    # Create a generated schema with a missing column and a dtype mismatch
    bad_gen = dict(generated_schema)  # shallow copy
    bad_gen["schema"] = dict(bad_gen["schema"])
    bad_gen["schema"]["columns"] = dict(bad_gen["schema"]["columns"])
    # remove 'maxhr'
    bad_gen["schema"]["columns"].pop("maxhr", None)
    # change dtype for 'age'
    bad_gen["schema"]["columns"]["age"] = "float64"

    res2 = dv.validate_schema(bad_gen, predefined_schema)
    assert res2["status"] == "FAILED"
    assert "maxhr" in res2["missing_columns"]
    assert "age" in res2["dtype_mismatches"]


def test_validate_domain_ranges_pass_and_fail(data_validation_config, ingestion_artifact):
    """
    validate_domain_ranges should PASS when generated ranges fall within predefined constraints,
    and FAIL when generated min/max violate constraints or missing keys.
    """
    dv = DataValidation(ingestion_artifact, data_validation_config)
    gen = dv.generated_schema
    pre = dv.predefined_schema

    # Sanity: current generated ranges are within predefined -> PASS
    domain_report = dv.validate_domain_ranges(gen, pre)
    assert domain_report["status"] == "PASSED"
    assert isinstance(domain_report["numeric_checks"], dict)

    # Modify generated schema to violate 'age' (found_min < expected_min)
    bad_gen = dict(gen)
    bad_gen["schema"] = dict(bad_gen["schema"])
    bad_gen["schema"]["domain_range"] = dict(bad_gen["schema"].get("domain_range", {}))
    bad_gen["schema"]["domain_range"]["age"] = {"min": -10.0, "max": 30.0}  # min too small

    report2 = dv.validate_domain_ranges(bad_gen, pre)
    assert report2["status"] == "FAILED"
    assert "age" in report2["numeric_checks"]
    assert report2["numeric_checks"]["age"]["valid"] is False

    # Missing domain key case: remove cholesterol from generated ranges
    bad_gen2 = dict(gen)
    bad_gen2["schema"] = dict(bad_gen2["schema"])
    bad_gen2["schema"]["domain_range"] = dict(bad_gen2["schema"].get("domain_range", {}))
    bad_gen2["schema"]["domain_range"].pop("cholesterol", None)

    report3 = dv.validate_domain_ranges(bad_gen2, pre)
    assert report3["status"] == "FAILED"
    assert "cholesterol" in report3["numeric_checks"]
    assert report3["numeric_checks"]["cholesterol"]["valid"] is False


def test_validate_categorical_values_pass_and_fail(data_validation_config, ingestion_artifact):
    """
    validate_categorical_values should PASS when generated categorical values are subset of allowed,
    and FAIL when unexpected categorical values are present.
    """
    dv = DataValidation(ingestion_artifact, data_validation_config)
    gen = dv.generated_schema
    pre = dv.predefined_schema

    # Sanity: current generated categories are subset -> PASS
    cat_report = dv.validate_categorical_values(gen, pre)
    assert cat_report["status"] == "PASSED"
    assert "categorical_checks" in cat_report

    # Insert an invalid categorical value in generated schema for 'sex'
    bad_gen = dict(gen)
    bad_gen["schema"] = dict(bad_gen["schema"])
    bad_gen["schema"]["unique_categorical_values"] = dict(bad_gen["schema"].get("unique_categorical_values", {}))
    bad_gen["schema"]["unique_categorical_values"]["sex"] = ["M", "Unknown"]

    cat_report2 = dv.validate_categorical_values(bad_gen, pre)
    assert cat_report2["status"] == "FAILED"
    assert cat_report2["categorical_checks"]["sex"]["valid"] is False
    assert "Unknown" in cat_report2["categorical_checks"]["sex"]["invalid_values"]


def test_generate_report_writes_file_and_returns_boolean(data_validation_config, ingestion_artifact, tmp_path):
    """
    generate_report should write a JSON report to data_validation_config.report_file_path
    and return True only when all component statuses are PASSED.
    """
    # Prepare DataValidation with paths pointing inside tmp_path
    data_validation_config.data_validation_dir = str(tmp_path / "validation")
    data_validation_config.report_file_path = str(Path(data_validation_config.data_validation_dir) / "validation_report.json")

    dv = DataValidation(ingestion_artifact, data_validation_config)

    # Case 1: all PASSED -> expect True and report file exists
    schema_ok = {"status": "PASSED"}
    domain_ok = {"status": "PASSED"}
    categorical_ok = {"status": "PASSED"}

    ok = dv.generate_report(schema_ok, domain_ok, categorical_ok)
    assert ok is True
    report_path = Path(data_validation_config.report_file_path)
    assert report_path.exists()

    content = json.loads(report_path.read_text(encoding="utf-8"))
    assert "validation_report" in content
    assert content["validation_report"]["status"] == "PASSED"

    # Case 2: one FAILED -> expect False
    schema_bad = {"status": "FAILED"}
    domain_ok2 = {"status": "PASSED"}
    categorical_ok2 = {"status": "PASSED"}

    bad = dv.generate_report(schema_bad, domain_ok2, categorical_ok2)
    assert bad is False
    # File should exist and have FAILED status
    content2 = json.loads(report_path.read_text(encoding="utf-8"))
    assert content2["validation_report"]["status"] == "FAILED"