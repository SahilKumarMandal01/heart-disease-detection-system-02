# """
# tests/test_model_trainer.py

# Unit tests for src/components/model_trainer.py

# Notes:
# - External services (dagshub, mlflow) are mocked to avoid network calls.
# - RandomizedSearchCV is mocked for fast, deterministic tests.
# - File IO utilities (load_numpy_array_data, save_object, write_json_file) are mocked.
# """

# import os
# import sys
# from pathlib import Path
# from types import SimpleNamespace
# from unittest.mock import patch, MagicMock

# import numpy as np
# import pytest

# # Make project importable
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# from src.components.model_trainer import ModelTrainer
# from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
# from src.entity.config_entity import ModelTrainerConfig, TrainingPipelineConfig
# from src.exception import HeartDiseaseException


# # ---------------------------------------------------------------------------
# # Fixtures
# # ---------------------------------------------------------------------------
# @pytest.fixture
# def pipeline_config(tmp_path):
#     """Create a TrainingPipelineConfig with artifact_dir set to tmp_path for isolation."""
#     cfg = TrainingPipelineConfig()
#     cfg.artifact_dir = str(tmp_path)
#     return cfg


# @pytest.fixture
# def model_trainer_config(pipeline_config):
#     """Return a ModelTrainerConfig instance for tests (uses tmp artifact dir)."""
#     return ModelTrainerConfig(pipeline_config)


# @pytest.fixture
# def data_transformation_artifact(tmp_path):
#     """
#     Create a DataTransformationArtifact with temporary file paths.
#     These files will not actually be read because load_numpy_array_data is patched.
#     """
#     return DataTransformationArtifact(
#         transformer_file_path=str(tmp_path / "transformer.pkl"),
#         transformed_train_file_path=str(tmp_path / "train_transformed.npy"),
#         transformed_test_file_path=str(tmp_path / "test_transformed.npy"),
#         feature_names_file_path=str(tmp_path / "feature_names.json"),
#     )


# # ---------------------------------------------------------------------------
# # Test: Constructor validation (ensures dagshub/mlflow stubbing)
# # ---------------------------------------------------------------------------
# @patch("src.components.model_trainer.dagshub.init")
# @patch("src.components.model_trainer.mlflow.set_experiment")
# def test_constructor_valid_and_invalid(mock_set_exp, mock_dagshub_init, data_transformation_artifact, model_trainer_config):
#     # valid constructor should not raise
#     trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
#     assert isinstance(trainer, ModelTrainer)

#     # invalid types should raise HeartDiseaseException
#     with pytest.raises(HeartDiseaseException):
#         ModelTrainer("bad_artifact", model_trainer_config)

#     with pytest.raises(HeartDiseaseException):
#         ModelTrainer(data_transformation_artifact, "bad_config")


# # ---------------------------------------------------------------------------
# # Test: evaluate_models with a Fake RandomizedSearchCV
# # ---------------------------------------------------------------------------
# def test_evaluate_models_with_fake_search(monkeypatch):
#     """
#     Replace RandomizedSearchCV with a FakeSearch that sets best_estimator_ and best_params_
#     and provides predictable predictions. This allows testing the evaluation logic without
#     running an actual hyperparameter search.
#     """

#     # Create tiny synthetic dataset
#     X_train = np.array([[0.1, 1.0], [0.2, 0.9], [0.3, 0.8], [0.4, 0.7]])
#     y_train = np.array([0, 0, 1, 1])
#     X_test = X_train.copy()
#     y_test = y_train.copy()

#     # Create a simple "estimator" with fit and predict
#     class DummyEstimator:
#         def __init__(self):
#             self._params = {"dummy_param": 1}

#         def fit(self, X, y):
#             self._n_samples = len(y)
#             return self

#         def predict(self, X):
#             # Return alternating classes for determinism
#             return np.array([0 if i % 2 == 0 else 1 for i in range(len(X))])

#         def get_params(self):
#             return self._params

#     # Fake search wrapper that mimics RandomizedSearchCV interface used in code
#     class FakeSearch:
#         def __init__(self, estimator, param_distributions, cv, n_jobs, n_iter, random_state, scoring):
#             # capture the supplied estimator so we can return it as best_estimator_
#             self.estimator = estimator

#         def fit(self, X, y):
#             # pretend we fitted and set best_estimator_ and best_params_
#             self.best_estimator_ = self.estimator
#             self.best_params_ = {"fake_param": "value"}
#             return self

#     # Patch RandomizedSearchCV in the module under test
#     monkeypatch.setattr("src.components.model_trainer.RandomizedSearchCV", FakeSearch)

#     models = {"DummyModel": DummyEstimator()}
#     params = {"DummyModel": {"dummy_param": [1]}}

#     # Call the classmethod evaluate_models
#     model_report, fitted_models = ModelTrainer.evaluate_models(
#         X_train, y_train, X_test, y_test, models, params
#     )

#     # Verify return shapes and that keys exist
#     assert "DummyModel" in model_report
#     assert "train" in model_report["DummyModel"]
#     assert "test" in model_report["DummyModel"]
#     assert "best_params" in model_report["DummyModel"]

#     assert "DummyModel" in fitted_models
#     assert hasattr(fitted_models["DummyModel"], "predict")


# # ---------------------------------------------------------------------------
# # Test: train_model integrates evaluate_models, tracking and persistence
# # ---------------------------------------------------------------------------
# @patch("src.components.model_trainer.save_object")
# @patch.object(ModelTrainer, "track_mlflow")  # avoid MLflow network calls inside
# def test_train_model_saves_and_returns(mock_track_mlflow, mock_save_object, monkeypatch, data_transformation_artifact, model_trainer_config):
#     """
#     Patch evaluate_models to return a controlled model_report and fitted_models.
#     Verify that train_model selects the best model and calls save_object.
#     """

#     # Create trainer instance
#     trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)

#     # Prepare fake X/y arrays
#     X_train = np.array([[0.1, 1.0], [0.2, 0.9], [0.3, 0.8], [0.4, 0.7]])
#     y_train = np.array([0, 0, 1, 1])
#     X_test = X_train.copy()
#     y_test = y_train.copy()

#     # Create a fake best model with get_params and predict
#     class BestModel:
#         def __init__(self):
#             pass

#         def predict(self, X):
#             return np.zeros(X.shape[0], dtype=int)

#         def get_params(self):
#             return {"param": "value"}

#     fake_best = BestModel()

#     # Create model_report such that 'GradientBoostingClassifier' becomes the best model
#     fake_model_report = {
#         "GradientBoostingClassifier": {
#             "train": {"accuracy": 1.0, "macro avg": {"precision": 1.0}},
#             "test": {"accuracy": 1.0, "macro avg": {"precision": 1.0}},
#             "best_params": {"n_estimators": 10}
#         }
#     }
#     fake_fitted = {"GradientBoostingClassifier": fake_best}

#     # Patch evaluate_models to return our fake objects
#     monkeypatch.setattr(
#         "src.components.model_trainer.ModelTrainer.evaluate_models",
#         staticmethod(lambda X_tr, y_tr, X_te, y_te, models, params: (fake_model_report, fake_fitted))
#     )

#     # Call train_model
#     best_model, model_report = trainer.train_model(X_train, y_train, X_test, y_test)

#     # verify returns and side-effects
#     assert best_model is fake_best
#     assert "GradientBoostingClassifier" in model_report
#     # save_object should be called at least once (saves final model & model_file_path)
#     assert mock_save_object.called


# # ---------------------------------------------------------------------------
# # Test: initiate_model_trainer orchestration (loads arrays, trains, writes metrics)
# # ---------------------------------------------------------------------------
# @patch("src.components.model_trainer.write_json_file")
# @patch("src.components.model_trainer.save_object")
# @patch("src.components.model_trainer.load_numpy_array_data")
# def test_initiate_model_trainer_full_flow(mock_load_np, mock_save_obj, mock_write_json, data_transformation_artifact, model_trainer_config, monkeypatch):
#     """
#     Tests that initiate_model_trainer:
#     - loads train/test arrays using load_numpy_array_data
#     - calls train_model and returns ModelTrainerArtifact
#     """

#     # Build fake transformed arrays with last column as integer target
#     train_arr = np.array([
#         [0.1, 0.2, 0],
#         [0.3, 0.4, 1],
#         [0.5, 0.6, 0],
#         [0.7, 0.8, 1],
#     ])
#     test_arr = train_arr.copy()

#     mock_load_np.side_effect = [train_arr, test_arr]

#     # Create a trainer and patch its train_model method to return a fake model & report
#     trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)

#     fake_model = MagicMock()
#     fake_report = {"GradientBoostingClassifier": {"test": {"macro avg": {"precision": 0.9}}}}

#     monkeypatch.setattr(trainer, "train_model", lambda X_tr, y_tr, X_te, y_te: (fake_model, fake_report))

#     artifact = trainer.initiate_model_trainer()

#     assert isinstance(artifact, ModelTrainerArtifact)
#     assert artifact.trained_model_file_path == model_trainer_config.model_file_path
#     assert artifact.trained_model_metrics_file_path == model_trainer_config.metrics_file_path
#     # write_json_file must have been called to save metrics
#     assert mock_write_json.called


# # ---------------------------------------------------------------------------
# # Test: initiate_model_trainer should raise on invalid shapes
# # ---------------------------------------------------------------------------
# @patch("src.components.model_trainer.load_numpy_array_data")
# def test_initiate_model_trainer_invalid_shape(mock_load_np, data_transformation_artifact, model_trainer_config):
#     # Provide 1D arrays (invalid)
#     mock_load_np.side_effect = [np.array([1, 2, 3]), np.array([1, 2, 3])]

#     trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)

#     with pytest.raises(HeartDiseaseException):
#         trainer.initiate_model_trainer()







"""
Testing strategy:
- Mock external services (dagshub, mlflow) to avoid network / side-effects.
- Mock RandomizedSearchCV so model evaluation is fast and deterministic.
- Mock file-writing utilities (save_object, write_json_file).
- Minimal strictness: verify return types, keys and that save/track functions are called.
"""

import numpy as np
import pytest, os, sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# Make project importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import module under test
import src.components.model_trainer as model_trainer_module
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import TrainingPipelineConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

# -----------------------------------------------------------------------------
# Helpers / Fake objects
# -----------------------------------------------------------------------------

class _FakeEstimator:
    """A minimal estimator that supports get_params and predict (no real training)."""

    def __init__(self, predict_value: int = 0):
        self._predict_value = predict_value

    def get_params(self):
        # Return a small, serializable dict similar to sklearn estimators.
        return {"dummy_param": 1}

    def predict(self, X):
        # Return a constant label (0 or 1) matching number of rows in X
        n = X.shape[0]
        return np.full(n, self._predict_value, dtype=int)


class _FakeSearch:
    """
    Fake replacement for RandomizedSearchCV used in tests.

    It accepts the same constructor signature but does not perform fitting.
    Instead, .fit(...) sets best_estimator_ to a provided fake estimator (or the passed estimator)
    and best_params_ to a deterministic dict.
    """

    def __init__(self, estimator=None, param_distributions=None, **kwargs):
        self.estimator = estimator
        self.param_distributions = param_distributions or {}
        self.best_estimator_ = None
        self.best_params_ = {}

    def fit(self, X, y):
        # Do not call underlying estimator.fit to keep tests fast.
        # Create a simple estimator that predicts the majority class seen in y.
        vals, counts = np.unique(y, return_counts=True)
        maj = int(vals[np.argmax(counts)])
        self.best_estimator_ = _FakeEstimator(predict_value=maj)
        self.best_params_ = {"mock_param": "mock_value"}
        return self


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_dagshub_and_mlflow(monkeypatch):
    """
    Replace dagshub.init and mlflow methods with no-op or MagicMock so tests do not
    attempt network access or create experiments.
    """
    # dagshub.init
    monkeypatch.setattr(model_trainer_module, "dagshub", SimpleNamespace(init=lambda *a, **k: None))

    # minimal mlflow surface
    fake_mlflow = SimpleNamespace()
    fake_mlflow.set_experiment = MagicMock()
    fake_mlflow.start_run = MagicMock()  # used as context manager in production; we provide a context manager below

    # create a context manager for start_run that is usable with 'with'
    class _FakeRunCM:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

    fake_mlflow.start_run = lambda *a, **k: _FakeRunCM()
    fake_mlflow.log_param = MagicMock()
    fake_mlflow.log_params = MagicMock()
    fake_mlflow.log_metric = MagicMock()
    fake_mlflow.get_tracking_uri = MagicMock(return_value="file:///tmp/mlruns")
    fake_mlflow.sklearn = SimpleNamespace(log_model=MagicMock())

    monkeypatch.setattr(model_trainer_module, "mlflow", fake_mlflow)
    yield


@pytest.fixture(autouse=True)
def patch_randomized_search(monkeypatch):
    """
    Patch RandomizedSearchCV used in model_trainer_module to the lightweight _FakeSearch,
    so evaluate_models runs quickly and deterministically.
    """
    monkeypatch.setattr(model_trainer_module, "RandomizedSearchCV", _FakeSearch)
    yield


@pytest.fixture
def fake_transformation_artifact(tmp_path):
    """
    Create a DataTransformationArtifact with paths (not used on-disk in these tests).
    """
    return DataTransformationArtifact(
        transformer_file_path=str(tmp_path / "transformer.pkl"),
        transformed_train_file_path=str(tmp_path / "train_transformed.npy"),
        transformed_test_file_path=str(tmp_path / "test_transformed.npy"),
        feature_names_file_path=str(tmp_path / "feature_names.json"),
    )


@pytest.fixture
def model_trainer_config(tmp_path):
    tp = TrainingPipelineConfig()
    cfg = ModelTrainerConfig(tp)
    # Override file paths to tmp (no actual writes, but keep values realistic)
    cfg.model_file_path = str(tmp_path / "model.pkl")
    cfg.metrics_file_path = str(tmp_path / "metrics.json")
    return cfg


@pytest.fixture(autouse=True)
def patch_save_and_io(monkeypatch):
    """
    Mock file-writing utilities: save_object, write_json_file, and load_numpy_array_data.
    """
    monkeypatch.setattr(model_trainer_module, "save_object", MagicMock())
    monkeypatch.setattr(model_trainer_module, "write_json_file", MagicMock())
    # load_numpy_array_data will be patched selectively in tests that need it
    yield


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_evaluate_models_returns_report_and_fitted_models():
    """
    evaluate_models should return:
      - model_report: dict containing 'train' and 'test' classification reports and 'best_params'
      - fitted_models: dict mapping model name to a fitted estimator (FakeEstimator)
    """

    # Create tiny synthetic data (2 classes)
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[1.5], [3.5]])
    y_test = np.array([0, 1])

    models = {"FakeModel": _FakeEstimator()}
    params = {"FakeModel": {"dummy_param": [1]}}

    model_report, fitted_models = ModelTrainer.evaluate_models(
        X_train, y_train, X_test, y_test, models, params
    )

    # Basic assertions (minimal strictness)
    assert isinstance(model_report, dict)
    assert isinstance(fitted_models, dict)
    assert "FakeModel" in model_report
    assert "FakeModel" in fitted_models

    rep = model_report["FakeModel"]
    assert "train" in rep and "test" in rep and "best_params" in rep
    assert isinstance(rep["train"], dict)
    assert isinstance(rep["test"], dict)


def test_train_model_selects_and_saves(monkeypatch, fake_transformation_artifact, model_trainer_config):
    """
    train_model should call evaluate_models (we exercise the real evaluate_models patched earlier),
    call track_mlflow (mocked), and call save_object to persist the selected model.
    """

    trainer = ModelTrainer(fake_transformation_artifact, model_trainer_config)

    # Patch ModelTrainer.track_mlflow so it does not attempt mlflow ops in tests
    monkeypatch.setattr(trainer, "track_mlflow", MagicMock())

    # Small dataset for training
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[1.5], [3.5]])
    y_test = np.array([0, 1])

    best_model, model_report = trainer.train_model(X_train, y_train, X_test, y_test)

    # Assertions (minimal)
    assert best_model is not None
    assert isinstance(model_report, dict)
    # verify save_object was called to persist the chosen model
    assert model_trainer_module.save_object.called


def test_initiate_model_trainer_reads_arrays_and_returns_artifact(monkeypatch, fake_transformation_artifact, model_trainer_config, tmp_path):
    """
    initiate_model_trainer should:
      - call load_numpy_array_data to obtain train/test arrays
      - call train_model (we'll patch it) and write metrics via write_json_file (mocked)
      - return a ModelTrainerArtifact with configured paths
    """
    # build synthetic train/test arrays with last column as target
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([0, 0, 1, 1])
    train_arr = np.c_[X_train, y_train]

    X_test = np.array([[1.5], [3.5]])
    y_test = np.array([0, 1])
    test_arr = np.c_[X_test, y_test]

    # Patch load_numpy_array_data to return our arrays
    monkeypatch.setattr(model_trainer_module, "load_numpy_array_data", lambda path: train_arr if "train" in path else test_arr)

    trainer = ModelTrainer(fake_transformation_artifact, model_trainer_config)

    # Patch train_model to return a dummy model and a simple model_report
    dummy_model = _FakeEstimator(predict_value=1)
    dummy_report = {"DummyModel": {"test": {"macro avg": {"precision": 0.5}}}}
    monkeypatch.setattr(trainer, "train_model", lambda X_tr, y_tr, X_te, y_te: (dummy_model, dummy_report))

    artifact = trainer.initiate_model_trainer()

    assert isinstance(artifact, ModelTrainerArtifact)
    assert artifact.trained_model_metrics_file_path == model_trainer_config.metrics_file_path
    # write_json_file is patched, but should have been called
    assert model_trainer_module.write_json_file.called
