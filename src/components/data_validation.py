import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

from src.exception import HeartDiseaseException
from src.logging import logging

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig

from src.utils.main_utils.utils import read_json_file, write_json_file


class DataValidation:
    """
    Validate generated dataset schema against a predefined schema.

    Validations performed:
      - Column presence and dtype matching
      - Numeric domain (min/max) checks
      - Categorical allowed-values checks
      - Writes a JSON validation report and returns a DataValidationArtifact
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ) -> None:
        try:
            logging.info("🔧 Initializing DataValidation component...", extra={
                "generated_schema": getattr(data_ingestion_artifact, "generated_schema", None),
                "predefined_schema": getattr(data_validation_config, "predefined_schema", None),
            })

            if not isinstance(data_ingestion_artifact, DataIngestionArtifact):
                raise TypeError("data_ingestion_artifact must be a DataIngestionArtifact instance.")

            if not isinstance(data_validation_config, DataValidationConfig):
                raise TypeError("data_validation_config must be a DataValidationConfig instance.")

            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            # Load JSON schemas (these util functions should raise if file missing or bad format)
            self.generated_schema: Dict[str, Any] = read_json_file(self.data_ingestion_artifact.generated_schema)
            self.predefined_schema: Dict[str, Any] = read_json_file(self.data_validation_config.predefined_schema)

            # Basic structural validation of loaded schemas
            self._assert_schema_structure(self.generated_schema, "generated_schema")
            self._assert_schema_structure(self.predefined_schema, "predefined_schema", predefined=True)

            logging.info("✅ DataValidation initialization complete.", extra={"status": "initialized"})

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # --------------------------
    # Helpers
    # --------------------------
    def _assert_schema_structure(self, schema: Dict[str, Any], name: str, predefined: bool = False) -> None:
        """
        Basic checks to ensure `schema` contains the expected top-level structure.
        For predefined schema, expects additional keys like 'domain_constraints' and 'categorical_allowed'.
        """
        try:
            if not isinstance(schema, dict):
                raise ValueError(f"{name} must be a dict (loaded JSON).")

            if "schema" not in schema or not isinstance(schema["schema"], dict):
                raise ValueError(f"{name} missing top-level 'schema' key or it's not an object.")

            if "columns" not in schema["schema"] or not isinstance(schema["schema"]["columns"], dict):
                raise ValueError(f"{name} must contain 'schema.columns' mapping.")

            if predefined:
                # predefined schema should include domain constraints and categorical allowed lists
                if "domain_constraints" not in schema["schema"]:
                    raise ValueError(f"predefined_schema missing 'schema.domain_constraints'.")
                if "categorical_allowed" not in schema["schema"]:
                    raise ValueError(f"predefined_schema missing 'schema.categorical_allowed'.")
        except Exception:
            raise

    # --------------------------
    # Schema Validation
    # --------------------------
    def validate_schema(self, generated_schema: Dict[str, Any], predefined_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that columns and dtypes match the predefined schema.

        Returns:
            dict with keys: status (PASSED|FAILED), missing_columns, unexpected_columns, dtype_mismatches
        """
        try:
            logging.info("🔎 Running schema validation...")

            expected_columns: Dict[str, str] = predefined_schema["schema"]["columns"]
            actual_columns: Dict[str, str] = generated_schema["schema"]["columns"]

            missing_columns = [c for c in expected_columns if c not in actual_columns]
            unexpected_columns = [c for c in actual_columns if c not in expected_columns]

            dtype_mismatches: Dict[str, Dict[str, str]] = {}
            for col, expected_dtype in expected_columns.items():
                if col in actual_columns:
                    found_dtype = actual_columns[col]
                    if expected_dtype != found_dtype:
                        dtype_mismatches[col] = {"expected": expected_dtype, "found": found_dtype}

            status = "PASSED" if (not missing_columns and not unexpected_columns and not dtype_mismatches) else "FAILED"

            result = {
                "status": status,
                "missing_columns": missing_columns,
                "unexpected_columns": unexpected_columns,
                "dtype_mismatches": dtype_mismatches,
            }

            logging.info("🧾 Schema validation completed.", extra={"schema_result": result})
            return result

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # --------------------------
    # Domain Range Validation
    # --------------------------
    def validate_domain_ranges(self, generated_schema: Dict[str, Any], predefined_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate numeric variable min/max constraints.

        Uses:
            generated_schema["schema"]["domain_range"]  (min/max from actual data)
            predefined_schema["schema"]["domain_constraints"] (expected min/max)
        """
        try:
            logging.info("📏 Validating numeric domain ranges...")

            gen_ranges: Dict[str, Dict[str, float]] = generated_schema["schema"].get("domain_range", {})
            pre_ranges: Dict[str, Dict[str, float]] = predefined_schema["schema"].get("domain_constraints", {})

            results: Dict[str, Any] = {}
            overall_status = "PASSED"

            for col, expected in pre_ranges.items():
                if col not in gen_ranges:
                    results[col] = {"valid": False, "reason": "missing_from_generated_schema"}
                    overall_status = "FAILED"
                    continue

                found = gen_ranges[col]
                # Guard against malformed values
                try:
                    found_min = float(found.get("min", float("nan")))
                    found_max = float(found.get("max", float("nan")))
                    expected_min = float(expected.get("min", float("-inf")))
                    expected_max = float(expected.get("max", float("inf")))
                except Exception:
                    results[col] = {"valid": False, "reason": "non_numeric_range_values", "expected": expected, "found": found}
                    overall_status = "FAILED"
                    continue

                is_valid = (found_min >= expected_min) and (found_max <= expected_max)

                if not is_valid:
                    overall_status = "FAILED"

                results[col] = {
                    "expected": {"min": expected_min, "max": expected_max},
                    "found": {"min": found_min, "max": found_max},
                    "valid": bool(is_valid),
                }

            report = {"status": overall_status, "numeric_checks": results}
            logging.info("📊 Domain range validation completed.", extra={"domain_report": report})
            return report

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # --------------------------
    # Categorical Values Validation
    # --------------------------
    def validate_categorical_values(self, generated_schema: Dict[str, Any], predefined_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify categorical features contain only allowed values.

        Uses:
            generated_schema["schema"]["unique_categorical_values"]
            predefined_schema["schema"]["categorical_allowed"]
        """
        try:
            logging.info("🧾 Validating categorical values...")

            gen_categories: Dict[str, list] = generated_schema["schema"].get("unique_categorical_values", {})
            pre_categories: Dict[str, list] = predefined_schema["schema"].get("categorical_allowed", {})

            results: Dict[str, Any] = {}
            overall_status = "PASSED"

            for col, allowed_values in pre_categories.items():
                actual_values = gen_categories.get(col, [])
                # Normalize to lists (defensive)
                allowed = list(allowed_values) if allowed_values is not None else []
                actual = list(actual_values) if actual_values is not None else []

                invalid_values = [v for v in actual if v not in allowed]

                valid = len(invalid_values) == 0
                if not valid:
                    overall_status = "FAILED"

                results[col] = {
                    "expected": allowed,
                    "found": actual,
                    "invalid_values": invalid_values,
                    "valid": valid,
                }

            report = {"status": overall_status, "categorical_checks": results}
            logging.info("✅ Categorical validation completed.", extra={"categorical_report": report})
            return report

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # --------------------------
    # Report generation
    # --------------------------
    def generate_report(self, schema_results: Dict[str, Any], domain_results: Dict[str, Any], categorical_results: Dict[str, Any]) -> bool:
        """
        Create a JSON validation report and persist it.

        Returns:
            True if overall validation PASSED, else False.
        """
        try:
            statuses = [schema_results.get("status"), domain_results.get("status"), categorical_results.get("status")]
            passed_count = statuses.count("PASSED")
            failed_count = statuses.count("FAILED")
            overall_status = "PASSED" if failed_count == 0 else "FAILED"

            report = {
                "validation_report": {
                    "status": overall_status,
                    "generated_at": datetime.now().isoformat(),
                    "summary": {
                        "total_checks": 3,
                        "passed": passed_count,
                        "failed": failed_count,
                    },
                    "schema_validation": schema_results,
                    "domain_validation": domain_results,
                    "categorical_validation": categorical_results,
                }
            }

            report_path = Path(self.data_validation_config.report_file_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_file(str(report_path), report)

            logging.info("📝 Validation report written.", extra={"report_path": str(report_path), "status": overall_status})
            return overall_status == "PASSED"

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # --------------------------
    # Orchestration
    # --------------------------
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Execute full validation pipeline and return DataValidationArtifact.
        """
        try:
            logging.info("🚀 Starting data validation workflow...")

            schema_results = self.validate_schema(self.generated_schema, self.predefined_schema)
            domain_results = self.validate_domain_ranges(self.generated_schema, self.predefined_schema)
            categorical_results = self.validate_categorical_values(self.generated_schema, self.predefined_schema)

            validation_status = self.generate_report(schema_results, domain_results, categorical_results)

            artifact = DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path,
                validation_status=validation_status,
            )

            logging.info("🎉 Data validation completed.", extra={"artifact": artifact.__dict__})
            return artifact

        except Exception as e:
            raise HeartDiseaseException(e, sys)
