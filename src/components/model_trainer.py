import sys
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from typing import Dict, Tuple, Any

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

from src.exception import HeartDiseaseException
from src.logging import logging
from src.utils.main_utils.utils import (
    load_numpy_array_data,
    save_object,
    write_json_file
)

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact
)


class ModelTrainer:
    """
    Handles ML model training, evaluation, hyper-param search,
    final model selection, saving metrics and artifacts.
    Includes DagsHub/MLflow experiment tracking.
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        try:
            logging.info("🚀 Initializing ModelTrainer...")

            if not isinstance(data_transformation_artifact, DataTransformationArtifact):
                raise TypeError("data_transformation_artifact must be a DataTransformationArtifact")

            if not isinstance(model_trainer_config, ModelTrainerConfig):
                raise TypeError("model_trainer_config must be a ModelTrainerConfig")

            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config

            # ---------------- DAGSHUB & MLFLOW SETUP ----------------
            logging.info("🔗 Initializing DagsHub & MLflow...")
            
            # Initialize DagsHub (Configures MLFLOW_TRACKING_URI & Auth)
            dagshub.init(
                repo_owner='thesahilmandal',
                repo_name='heart-disease-detection-system-02', 
                mlflow=True)
            
            # Set experiment name for organization
            mlflow.set_experiment("Heart Disease Final Train")
            # --------------------------------------------------------

            logging.info("✨ ModelTrainer initialized successfully.")

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ======================================================================
    def track_mlflow(self, best_model, classification_metrics: dict):
        """
        Logs the best model, its parameters, and metrics for BOTH classes to MLflow.
        """
        try:
            logging.info("📊 Starting MLflow tracking for the best model...")
            
            best_model_name = best_model.__class__.__name__
            
            # Start Run
            with mlflow.start_run():
                
                # -------------------------------------------------------
                # 1. Log Model Name & Hyperparameters
                # -------------------------------------------------------
                mlflow.log_param("model_name", best_model_name)
                mlflow.log_params(best_model.get_params())

                # -------------------------------------------------------
                # 2. Log Metrics (Split by Class)
                # -------------------------------------------------------
                test_metrics = classification_metrics['test']
                
                # --- Global Metrics ---
                mlflow.log_metric("overall_accuracy", test_metrics['accuracy'])
                mlflow.log_metric("macro_f1_score", test_metrics['macro avg']['f1-score'])

                # --- Class 0 Metrics (e.g., Healthy) ---
                if '0' in test_metrics:
                    mlflow.log_metric("class_0_precision", test_metrics['0']['precision'])
                    mlflow.log_metric("class_0_recall",    test_metrics['0']['recall'])
                    mlflow.log_metric("class_0_f1_score",  test_metrics['0']['f1-score'])

                # --- Class 1 Metrics (e.g., Heart Disease) ---
                if '1' in test_metrics:
                    mlflow.log_metric("class_1_precision", test_metrics['1']['precision'])
                    mlflow.log_metric("class_1_recall",    test_metrics['1']['recall'])
                    mlflow.log_metric("class_1_f1_score",  test_metrics['1']['f1-score'])

                # -------------------------------------------------------
                # 3. Log Model Artifact
                # -------------------------------------------------------
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        best_model, 
                        "model", 
                        registered_model_name=best_model_name
                    )
                else:
                    mlflow.sklearn.log_model(best_model, "model")
            
            logging.info("✅ MLflow tracking completed via DagsHub integration.")

        except Exception as e:
            logging.warning(f"❌ MLflow tracking failed: {e}")

    # ======================================================================
    @classmethod
    def evaluate_models(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        models: Dict[str, Any],
        params: Dict[str, Dict]
    ) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        try:
            logging.info("🧪 Starting model evaluation...")

            model_report = {}
            fitted_models = {}

            for name, model in models.items():
                logging.info(f"🔍 Evaluating model: {name}")

                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params.get(name, {}),
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    n_jobs=-1,
                    n_iter=1,
                    random_state=42,
                    scoring="accuracy"
                )

                search.fit(X_train, y_train)

                best_model = search.best_estimator_
                best_params = search.best_params_

                fitted_models[name] = best_model

                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Generate classification report as a dictionary
                model_report[name] = {
                    "train": classification_report(y_train, y_train_pred, output_dict=True),
                    "test": classification_report(y_test, y_test_pred, output_dict=True),
                    "best_params": best_params
                }

                logging.info(f"🟦 Completed evaluation for {name}")

            return model_report, fitted_models

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ======================================================================
    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray
    ):
        try:
            logging.info("🧠 Starting model training...")

            models = {
                "GradientBoostingClassifier": GradientBoostingClassifier()
            }

            # Hyperparameters for RandomizedSearchCV
            params = {
                "GradientBoostingClassifier": {
                    "learning_rate": [0.047454011884736254],
                    "max_depth": [3],
                    "max_features": [None],
                    "min_samples_leaf": [8],
                    "min_samples_split": [8],
                    "n_estimators": [221],
                    "subsample": [0.7467983561008608]
                }
            }

            model_report, fitted_models = self.evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            logging.info("🎯 Selecting best-performing model...")

            # Select model based on Macro Avg Precision on Test set
            score_dict = {
                name: metrics["test"]["macro avg"]["precision"]
                for name, metrics in model_report.items()
            }

            best_model_name = max(score_dict, key=score_dict.get)
            best_model = fitted_models[best_model_name]
            
            # ---------------- TRACK WITH DAGSHUB/MLFLOW ----------------
            self.track_mlflow(best_model, model_report[best_model_name])
            # -----------------------------------------------------------

            # Save "standard" model object
            save_object("final_model/model.pkl", best_model)
            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=best_model
            )

            logging.info(f"🏆 Best model: {best_model_name}")
            logging.info(f"🔧 Best hyperparameters: {model_report[best_model_name]['best_params']}")

            logging.info("🟩 Model training complete.")
            return best_model, model_report

        except Exception as e:
            raise HeartDiseaseException(e, sys)

    # ======================================================================
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("🚦 Starting ModelTrainer pipeline...")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            if train_arr.ndim != 2 or test_arr.ndim != 2:
                raise HeartDiseaseException("Invalid shape for transformed arrays.", sys)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

            # -------------------- CRITICAL FIX --------------------
            # Explicitly cast targets to int to ensure classification_report 
            # keys are '0' and '1' (strings of ints) and not '0.0' and '1.0'.
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            # ------------------------------------------------------

            final_model, model_report = self.train_model(X_train, y_train, X_test, y_test)

            write_json_file(
                file_path=self.model_trainer_config.metrics_file_path,
                content=model_report
            )

            # NOTE: model is saved inside train_model, but we define the artifact path here
            artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.model_file_path,
                trained_model_metrics_file_path=self.model_trainer_config.metrics_file_path
            )

            logging.info("🎉 ModelTrainer pipeline completed successfully!")

            return artifact

        except Exception as e:
            raise HeartDiseaseException(e, sys)