import os
from datetime import datetime
from src.constant import training_pipeline


class TrainingPipelineConfig:
    """
    Configuration for the overall training pipeline.
    Creates a timestamped artifact directory that all pipeline components use.
    """

    def __init__(self, timestamp: datetime = None):
        timestamp = timestamp or datetime.now()
        formatted_timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        # Core pipeline metadata
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_name: str = training_pipeline.ARTIFACT_DIR
        self.model_dir = os.path.join("final_model")
        # Main artifact directory for this pipeline run
        self.artifact_dir: str = os.path.join(self.artifact_name, formatted_timestamp)
        self.timestamp: str = formatted_timestamp


class DataIngestionConfig:
    """
    Configuration for the Data Ingestion component.
    Defines:
    - Output directories (raw, train, test)
    - Schema file path
    - MongoDB extraction settings
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        artifact_dir = training_pipeline_config.artifact_dir

        # Root directory for ingestion artifacts
        self.data_ingestion_dir: str = os.path.join(
            artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )

        # File paths for ingestion artifacts
        self.raw_data_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_RAW_DATA
        )
        self.train_data_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_TRAIN_DATA
        )
        self.test_data_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_TEST_DATA
        )
        self.schema_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_SCHEMA
        )

        # Database and split configuration
        self.train_test_split_ratio: float = (
            training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    """
    Configuration for the Data Validation component.
    Stores:
    - Validation directory
    - Validation report path
    - Predefined schema reference
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        artifact_dir = training_pipeline_config.artifact_dir

        self.data_validation_dir: str = os.path.join(
            artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_REPORT
        )

        # Provided schema reference
        self.predefined_schema: str = training_pipeline.PREDEFINED_SCHEMA


class DataTransformationConfig:
    """
    Configuration for the Data Transformation component.
    Stores:
    - Transformer path
    - Train/test transformed data paths
    - Feature names file path
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        artifact_dir = training_pipeline_config.artifact_dir

        self.data_transformation_dir: str = os.path.join(
            artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        self.transformer: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMER
        )
        self.training_data: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRAINING_DATA
        )
        self.testing_data: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TESTING_DATA
        )
        self.feature_names: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_FEATURE_NAMES
        )


class ModelTrainerConfig: 
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        artifact_dif = training_pipeline_config.artifact_dir
        self.model_trainer_dir = os.path.join(
            artifact_dif,
            training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.model_file_path = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_MODEL_FILE_PATH
        )
        self.metrics_file_path = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_METRICS
        )