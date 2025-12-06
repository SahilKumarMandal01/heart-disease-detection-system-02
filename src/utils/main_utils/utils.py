import os
import sys
import yaml
import pickle
import json
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

from src.exception import HeartDiseaseException
from src.logging import logging


# -------------------------------------------------------------------
# INTERNAL HELPERS
# -------------------------------------------------------------------

def _prepare_file_path(file_path: str, replace: bool = True) -> None:
    """
    Ensure the directory for file_path exists.
    Optionally remove the existing file before writing.
    """
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        if replace and os.path.isfile(file_path):
            logging.info(f"Removing existing file at: {file_path}")
            os.remove(file_path)

    except Exception as e:
        logging.error(f"Failed to prepare file path: {file_path}")
        raise HeartDiseaseException(e, sys)


# -------------------------------------------------------------------
# YAML UTILITIES
# -------------------------------------------------------------------

def read_yaml_file(file_path: str) -> Dict[str, Any]:
    """Load a YAML file and return the parsed dictionary."""
    try:
        logging.info(f"Reading YAML file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file) or {}

    except Exception as e:
        logging.error(f"Error reading YAML file: {file_path}")
        raise HeartDiseaseException(e, sys)


def write_yaml_file(file_path: str, content: Any, replace: bool = True) -> None:
    """Write a Python object to a YAML file."""
    try:
        logging.info(f"Writing YAML file: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(content, yaml_file, sort_keys=False)

    except Exception as e:
        logging.error(f"Error writing YAML file: {file_path}")
        raise HeartDiseaseException(e, sys)


# -------------------------------------------------------------------
# NUMPY UTILITIES
# -------------------------------------------------------------------

def save_numpy_array_data(file_path: str, array: np.ndarray, replace: bool = True) -> None:
    """Persist a NumPy array to disk."""
    try:
        logging.info(f"Saving NumPy array: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        logging.error(f"Error saving NumPy array: {file_path}")
        raise HeartDiseaseException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """Load a NumPy array from disk."""
    try:
        logging.info(f"Loading NumPy array: {file_path}")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=False)

    except Exception as e:
        logging.error(f"Error loading NumPy array: {file_path}")
        raise HeartDiseaseException(e, sys)


# -------------------------------------------------------------------
# PICKLE UTILITIES
# -------------------------------------------------------------------

def save_object(file_path: str, obj: Any, replace: bool = True) -> None:
    """Serialize and save any Python object using pickle."""
    try:
        logging.info(f"Saving pickled object: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        logging.error(f"Error saving pickled object: {file_path}")
        raise HeartDiseaseException(e, sys)


def load_object(file_path: str) -> Any:
    """Load a Python object serialized via pickle."""
    try:
        logging.info(f"Loading pickled object: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.error(f"Error loading pickled object: {file_path}")
        raise HeartDiseaseException(e, sys)


# -------------------------------------------------------------------
# JSON UTILITIES
# -------------------------------------------------------------------

def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file and return a dictionary.
    Returns {} if the JSON file is empty.
    """
    try:
        logging.info(f"Reading JSON file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    except json.JSONDecodeError:
        logging.warning(f"JSON file empty or malformed, returning empty dict: {file_path}")
        return {}

    except Exception as e:
        logging.error(f"Error reading JSON file: {file_path}")
        raise HeartDiseaseException(e, sys)


def write_json_file(file_path: str, content: Any, replace: bool = True) -> None:
    """
    Write a Python object to JSON on disk.
    Automatically formats with indentation for readability.
    """
    try:
        logging.info(f"Writing JSON file: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "w", encoding="utf-8") as file_obj:
            json.dump(content, file_obj, indent=4, ensure_ascii=False)

    except Exception as e:
        logging.error(f"Error writing JSON file: {file_path}")
        raise HeartDiseaseException(e, sys)