from __future__ import annotations

import sys
from typing import Any, Callable, Optional

# Pipeline components
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Config entities
from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

# Core utilities
from src.exception import HeartDiseaseException
from src.logging import logging
from src.constant.training_pipeline import TRAINING_BUCKET_NAME
from src.cloud.s3_syncer import S3Sync


class TrainingPipeline:
    """
    High-level orchestration of the ML training pipeline.
    """

    def __init__(self, config: Optional[TrainingPipelineConfig] = None) -> None:
        try:
            logging.info("⚙️  Initializing Training Pipeline...")

            self.pipeline_config = config if config is not None else TrainingPipelineConfig()
            self.s3_sync = S3Sync()

        except Exception as e:
            logging.exception("❌ Error initializing pipeline.")
            raise HeartDiseaseException(e, sys) from e

    # ----------------------------------------------------------------------
    # Centralized Step Executor (Handles all logging)
    # ----------------------------------------------------------------------
    @staticmethod
    def _execute_step(
        step_name: str, 
        step_fn: Callable[..., Any], 
        *args, 
        **kwargs
    ) -> Any:
        """
        Executes a step and handles standard logging centrally.
        """
        try:
            logging.info(f"⏩ Running step: {step_name}")
            print(f"⏩ Running step: {step_name}")
            
            
            result = step_fn(*args, **kwargs)
            
            logging.info(f"✅ Completed step: {step_name}")
            logging.info(result)
            print(f"✅ Completed step: {step_name}")
            print(result)
            
            return result

        except Exception as e:
            logging.exception(f"❌ Failed step: {step_name}")
            raise HeartDiseaseException(e, sys) from e

    # ----------------------------------------------------------------------
    # Pipeline Stages (Cleaned: Logic Only, No Redundant Logs)
    # ----------------------------------------------------------------------
    def start_data_ingestion(self) -> Any:
        try:
            cfg = DataIngestionConfig(self.pipeline_config)
            ingestion = DataIngestion(cfg)
            return ingestion.initiate_data_ingestion()
        except Exception as e:
            raise HeartDiseaseException(e, sys) from e

    def start_data_validation(self, ingestion_artifact: Any) -> Any:
        try:
            cfg = DataValidationConfig(self.pipeline_config)
            validator = DataValidation(ingestion_artifact, cfg)
            return validator.initiate_data_validation()
        except Exception as e:
            raise HeartDiseaseException(e, sys) from e

    def start_data_transformation(self, ingestion_artifact: Any, validation_artifact: Any) -> Any:
        try:
            cfg = DataTransformationConfig(self.pipeline_config)
            transformer = DataTransformation(ingestion_artifact, validation_artifact, cfg)
            return transformer.initiate_data_transformation()
        except Exception as e:
            raise HeartDiseaseException(e, sys) from e

    def start_model_training(self, transformation_artifact: Any) -> Any:
        try:
            cfg = ModelTrainerConfig(self.pipeline_config)
            trainer = ModelTrainer(transformation_artifact, cfg)
            return trainer.initiate_model_trainer()
        except Exception as e:
            raise HeartDiseaseException(e, sys) from e

    # ----------------------------------------------------------------------
    # S3 Sync Operations
    # ----------------------------------------------------------------------
    def sync_artifact_dir_to_s3(self) -> None:
        try:
            if not getattr(self.pipeline_config, "artifact_dir", None):
                return

            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.pipeline_config.timestamp}"
            
            self.s3_sync.sync_folder_to_s3(
                folder=self.pipeline_config.artifact_dir,
                aws_bucket_url=aws_bucket_url
            )

        except Exception as e:
            logging.exception("❌ Artifact sync failed.")
            raise HeartDiseaseException(e, sys) from e

    def sync_saved_model_dir_to_s3(self) -> None:
        try:
            if not getattr(self.pipeline_config, "model_dir", None):
                return

            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.pipeline_config.timestamp}"
            
            self.s3_sync.sync_folder_to_s3(
                folder=self.pipeline_config.model_dir,
                aws_bucket_url=aws_bucket_url
            )

        except Exception as e:
            logging.exception("❌ Model sync failed.")
            raise HeartDiseaseException(e, sys) from e

    # ----------------------------------------------------------------------
    # Main Execution
    # ----------------------------------------------------------------------
    def run_pipeline(self) -> Any:
        try:
            logging.info("🚀 Pipeline execution started\n")

            ingestion_artifact = self._execute_step("Data Ingestion", self.start_data_ingestion)
            
            validation_artifact = self._execute_step(
                "Data Validation", 
                self.start_data_validation, 
                ingestion_artifact
            )
            
            transformation_artifact = self._execute_step(
                "Data Transformation", 
                self.start_data_transformation, 
                ingestion_artifact, 
                validation_artifact
            )
            
            model_artifact = self._execute_step(
                "Model Training", 
                self.start_model_training, 
                transformation_artifact
            )

            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            logging.info("🏁 Pipeline execution completed successfully")
            return model_artifact

        except Exception as e:
            # Exception is already logged in _execute_step or sync methods
            raise HeartDiseaseException(e, sys) from e


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.exception("❌ Top-level pipeline failure.")
        raise