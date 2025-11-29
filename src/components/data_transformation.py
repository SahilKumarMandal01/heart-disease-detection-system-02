import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.exception import HeartDiseaseException
from src.logging import logging
from src.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object,
    write_json_file,
)


class DataTransformation:
    """
    Handles preprocessing for the Heart Disease pipeline:

      ✓ Reads ingested train/test datasets
      ✓ Performs feature engineering
      ✓ Builds preprocessing pipeline (ordinal + nominal + numeric)
      ✓ Applies transformations
      ✓ Saves transformed arrays & preprocessing object
      ✓ Saves feature names
      ✓ Returns DataTransformationArtifact
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            logging.info("🚀 Initializing DataTransformation component...")

            if not isinstance(data_ingestion_artifact, DataIngestionArtifact):
                raise TypeError("data_ingestion_artifact must be DataIngestionArtifact")
            if not isinstance(data_validation_artifact, DataValidationArtifact):
                raise TypeError("data_validation_artifact must be DataValidationArtifact")
            if not isinstance(data_transformation_config, DataTransformationConfig):
                raise TypeError("data_transformation_config must be DataTransformationConfig")

            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

            logging.info("✨ DataTransformation initialized.")

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ----------------------------------------------------------------------
    # Utility: Read CSV
    # ----------------------------------------------------------------------
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"📄 Loading dataset from: {file_path}")
            df = pd.read_csv(file_path)
            logging.info(f"📦 Loaded dataset: shape={df.shape}")
            return df
        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ----------------------------------------------------------------------
    # Utility: Outlier Clipping
    # ----------------------------------------------------------------------
    @staticmethod
    def fix_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame, column: str):
        try:
            logging.info(f"📉 Clipping outliers in: {column}")

            Q1, Q3 = train_df[column].quantile([0.25, 0.75])
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            train_df[column] = train_df[column].clip(lower, upper)
            test_df[column] = test_df[column].clip(lower, upper)

            return train_df, test_df

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ----------------------------------------------------------------------
    # Feature Engineering
    # ----------------------------------------------------------------------
    def feature_engineering(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            logging.info("🧠 Performing feature engineering...")

            # ---------------- RestingBP ----------------
            median_rbp = train_df["restingbp"].median()
            train_df["restingbp"] = train_df["restingbp"].fillna(median_rbp)
            test_df["restingbp"] = test_df["restingbp"].fillna(median_rbp)

            train_df, test_df = self.fix_outliers(train_df, test_df, "restingbp")

            # ---------------- Cholesterol ----------------
            median_chol_by_class = train_df.groupby("heartdisease")["cholesterol"].median()

            train_df["cholesterol"] = train_df.apply(
                lambda r: median_chol_by_class[r["heartdisease"]]
                if pd.isna(r["cholesterol"])
                else r["cholesterol"],
                axis=1,
            )

            test_df["cholesterol"] = test_df.apply(
                lambda r: (
                    median_chol_by_class[r["heartdisease"]]
                    if pd.isna(r["cholesterol"]) and r["heartdisease"] in median_chol_by_class.index
                    else (
                        median_chol_by_class.mean()
                        if pd.isna(r["cholesterol"])
                        else r["cholesterol"]
                    )
                ),
                axis=1,
            )

            train_df, test_df = self.fix_outliers(train_df, test_df, "cholesterol")

            # ---------------- MaxHR ----------------
            train_df, test_df = self.fix_outliers(train_df, test_df, "maxhr")

            # ---------------- Oldpeak Binning ----------------
            bins = [0, 1.0, 2.0, 4.0, float("inf")]
            labels = ["Normal", "Mild", "Moderate", "Severe"]

            train_df["oldpeak"] = pd.cut(
                train_df["oldpeak"], bins=bins, labels=labels,
                include_lowest=True, right=False
            )
            test_df["oldpeak"] = pd.cut(
                test_df["oldpeak"], bins=bins, labels=labels,
                include_lowest=True, right=False
            )

            logging.info("✨ Feature engineering done.")
            return train_df, test_df

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ----------------------------------------------------------------------
    # Preprocessing Pipeline
    # ----------------------------------------------------------------------
    def get_preprocessor(self) -> ColumnTransformer:
        try:
            logging.info("⚙️ Building preprocessing pipeline...")

            ordinal_features = ["st_slope", "oldpeak"]
            ordinal_categories = [
                ["Down", "Flat", "Up"],
                ["Normal", "Mild", "Moderate", "Severe"],
            ]

            nominal_features = ["sex", "chestpaintype", "restingecg", "exerciseangina"]
            numeric_features = ["age", "restingbp", "cholesterol", "fastingbs", "maxhr"]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("ordinal", OrdinalEncoder(categories=ordinal_categories), ordinal_features),
                    ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_features),
                    ("numeric", "passthrough", numeric_features),
                ]
            )

            logging.info("🧩 Preprocessing pipeline created.")
            return preprocessor, ordinal_features, nominal_features, numeric_features

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ----------------------------------------------------------------------
    # Orchestration
    # ----------------------------------------------------------------------
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("🚦 Starting Data Transformation Pipeline...")

        try:
            if not self.data_validation_artifact.validation_status:
                raise HeartDiseaseException(
                    "❌ Data validation failed. Aborting transformation.",
                    sys,
                )

            # ---------------- Load Data ----------------
            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # ---------------- Feature Engineering ----------------
            train_feat_df, test_feat_df = self.feature_engineering(
                train_df.copy(), test_df.copy()
            )

            # ---------------- Split Input / Target ----------------
            X_train, y_train = train_feat_df.drop("heartdisease", axis=1), train_feat_df["heartdisease"]
            X_test, y_test = test_feat_df.drop("heartdisease", axis=1), test_feat_df["heartdisease"]

            # ---------------- Preprocessor ----------------
            preprocessor, ordinal_f, nominal_f, numeric_f = self.get_preprocessor()
            fitted_preprocessor = preprocessor.fit(X_train)

            save_object(self.data_transformation_config.transformer, fitted_preprocessor)
            save_object("final_model/preprocessor.pkl", fitted_preprocessor)    # optional

            logging.info("🔄 Applying preprocessing transformations...")
            train_transformed = fitted_preprocessor.transform(X_train)
            test_transformed = fitted_preprocessor.transform(X_test)

            # ---------------- Feature Names ----------------
            logging.info("📝 Extracting feature names...")

            nominal_names = (
                fitted_preprocessor.named_transformers_["nominal"]
                .get_feature_names_out(nominal_f)
                .tolist()
            )

            feature_names = ordinal_f + nominal_names + numeric_f

            # ---------------- Combine Feature + Target ----------------
            train_arr = np.c_[train_transformed, y_train.values]
            test_arr = np.c_[test_transformed, y_test.values]

            # ---------------- Save Outputs ----------------
            logging.info("💾 Saving transformed arrays and metadata...")

            save_numpy_array_data(self.data_transformation_config.training_data, train_arr)
            save_numpy_array_data(self.data_transformation_config.testing_data, test_arr)
            write_json_file(self.data_transformation_config.feature_names, feature_names)

            logging.info("🎉 Data Transformation completed successfully!")

            # ---------------- Produce Artifact ----------------
            return DataTransformationArtifact(
                transformer_file_path=self.data_transformation_config.transformer,
                transformed_train_file_path=self.data_transformation_config.training_data,
                transformed_test_file_path=self.data_transformation_config.testing_data,
                feature_names_file_path=self.data_transformation_config.feature_names,
            )

        except Exception as e:
            raise HeartDiseaseException(e, sys)
