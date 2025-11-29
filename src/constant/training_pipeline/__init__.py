import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

"""
Defining common constant variable for training pipeline.
"""
PIPELINE_NAME: str = "HeartDisease"
ARTIFACT_DIR: str = 'Artifacts'
PREDEFINED_SCHEMA: str = "DataSchema/schema_predefined.json"
TRAINING_BUCKET_NAME: str = "heartdisease02"


"""
Data Ingestion related constant start with DATA_INGESTION var name
"""
DATA_INGESTION_COLLECTION_NAME: str = os.getenv("COLLECTION_NAME")
DATA_INGESTION_DATABASE_NAME: str = os.getenv("DATABASE_NAME")
DATA_INGESTION_DIR_NAME: str = "1_data_ingestion"
DATA_INGESTION_RAW_DATA: str = "heart_disease.csv"
DATA_INGESTION_TRAIN_DATA: str = "train.csv"
DATA_INGESTION_TEST_DATA: str = "test.csv"
DATA_INGESTION_SCHEMA: str = "schema_generated.json"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


"""
Data Validation related constant start with DATA_VALIDAION var name
"""
DATA_VALIDATION_DIR_NAME: str = '2_data_validation'
DATA_VALIDATION_REPORT: str = "validation_report.json"


"""
Data Transformation related constant start with DATA_TRANSFORMATION var name
"""
DATA_TRANSFORMATION_DIR_NAME: str = "3_data_transformation"
DATA_TRANSFORMATION_TRANSFORMER: str = "transformer.pkl"
DATA_TRANSFORMATION_TRAINING_DATA: str = "transformed_train.npy"
DATA_TRANSFORMATION_TESTING_DATA: str = "transformed_test.npy"
DATA_TRANSFORMATION_FEATURE_NAMES: str = "feature_names.json"


"""
Model Trainer related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "4_model_trainer"
MODEL_TRAINER_MODEL_FILE_PATH = "model.pkl"
MODEL_TRAINER_METRICS = "metrics.json"