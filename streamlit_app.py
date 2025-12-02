from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------
# Detect if running inside Docker
# ---------------------------
RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")

# ---------------------------
# Make project root importable
# ---------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Project-specific imports
from src.utils.main_utils.utils import load_object
from src.utils.ml_utils.model.estimator import HeartDiseaseModel

# Import training pipeline ONLY if not inside Docker
if not RUNNING_IN_DOCKER:
    from src.pipeline.training_pipeline import TrainingPipeline
else:
    TrainingPipeline = None


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------
# Constants / Schemas
# ---------------------------
CATEGORICAL_ALLOWED: Dict[str, List] = {
    "sex": ["M", "F"],
    "chestpaintype": ["ATA", "NAP", "ASY", "TA"],
    "restingecg": ["Normal", "ST", "LVH"],
    "exerciseangina": ["Y", "N"],
    "st_slope": ["Up", "Flat", "Down"],
    "fastingbs": [0, 1],
}

DOMAIN_CONSTRAINTS: Dict[str, Tuple[float, float]] = {
    "age": (0, 120),
    "restingbp": (1, 250),
    "cholesterol": (1, 700),
    "maxhr": (40, 250),
    "oldpeak": (0.0, 10.0),
}

DEFAULT_SAMPLE_ROWS = 5


# ---------------------------
# Model existence check
# ---------------------------
def model_exists() -> bool:
    model_path = os.path.join("final_model", "model.pkl")
    preproc_path = os.path.join("final_model", "preprocessor.pkl")
    return os.path.exists(model_path) and os.path.exists(preproc_path)


# ---------------------------
# Utilities
# ---------------------------
def cached_resource(func):
    return st.cache_resource(func)


@cached_resource
def load_prediction_service():
    """
    Load model + preprocessor and return the wrapped HeartDiseaseModel.
    """
    model_path = os.path.join("final_model", "model.pkl")
    preproc_path = os.path.join("final_model", "preprocessor.pkl")

    model = load_object(model_path)
    preprocessor = load_object(preproc_path)

    return HeartDiseaseModel(preprocessor=preprocessor, model=model)


# ---------------------------
# Train pipeline + reload
# ---------------------------
def train_pipeline_and_reload() -> Optional[HeartDiseaseModel]:
    if RUNNING_IN_DOCKER:
        st.error("Training is disabled inside Docker runtime.")
        return None

    try:
        st.info("🚀 Starting training pipeline... please wait.")
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()

        st.success("🎉 Training complete! Reloading model...")

        load_prediction_service.clear()  # Clear cache
        return load_prediction_service()

    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        return None


# ---------------------------
# Validation utilities
# ---------------------------
def validate_row(row: pd.Series) -> Optional[str]:
    for col, allowed in CATEGORICAL_ALLOWED.items():
        if col in row and pd.notna(row[col]):
            if row[col] not in allowed:
                return f"Column '{col}' has invalid value '{row[col]}'. Allowed: {allowed}"

    for col, (lo, hi) in DOMAIN_CONSTRAINTS.items():
        if col in row and pd.notna(row[col]):
            try:
                val = float(row[col])
            except Exception:
                return f"Column '{col}' must be numeric."
            if not (lo <= val <= hi):
                return f"Column '{col}' out of range [{lo}, {hi}]. Found {val}"

    return None


def validate_dataframe(df: pd.DataFrame, require_columns: bool = True) -> Tuple[bool, List[str]]:
    required_columns = [
        "age", "sex", "chestpaintype", "restingbp", "cholesterol",
        "fastingbs", "restingecg", "maxhr", "exerciseangina",
        "oldpeak", "st_slope",
    ]
    errors: List[str] = []

    if require_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
            return False, errors

    for i, row in df.iterrows():
        err = validate_row(row)
        if err:
            errors.append(f"Row {i}: {err}")
        if len(errors) > 10:
            errors.append("More than 10 validation errors; stopping further checks.")
            break

    return (len(errors) == 0), errors


def template_csv_bytes() -> bytes:
    sample = pd.DataFrame(
        [
            {
                "age": 58,
                "sex": "M",
                "chestpaintype": "ATA",
                "restingbp": 140,
                "cholesterol": 200,
                "fastingbs": 0,
                "restingecg": "Normal",
                "maxhr": 150,
                "exerciseangina": "N",
                "oldpeak": 0.0,
                "st_slope": "Up",
            }
        ]
        * DEFAULT_SAMPLE_ROWS
    )
    return sample.to_csv(index=False).encode("utf-8")


def human_label_from_pred(p: int) -> str:
    return "Heart Disease" if int(p) == 1 else "No Heart Disease"


# ---------------------------
# UI — Single Prediction
# ---------------------------
def single_prediction_ui(pred_service: HeartDiseaseModel):
    st.header("🔍 Predict Heart Disease (Single Entry)")
    with st.form("single_prediction_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([1, 1, 0.6])

        with c1:
            age = st.number_input("Age", int(DOMAIN_CONSTRAINTS["age"][0]), int(DOMAIN_CONSTRAINTS["age"][1]), value=50)
            sex = st.selectbox("Sex", CATEGORICAL_ALLOWED["sex"])
            chestpaintype = st.selectbox("Chest Pain Type", CATEGORICAL_ALLOWED["chestpaintype"])
            restingbp = st.number_input("Resting BP", int(DOMAIN_CONSTRAINTS["restingbp"][0]), int(DOMAIN_CONSTRAINTS["restingbp"][1]), value=120)
            cholesterol = st.number_input("Cholesterol", int(DOMAIN_CONSTRAINTS["cholesterol"][0]), int(DOMAIN_CONSTRAINTS["cholesterol"][1]), value=200)

        with c2:
            fastingbs = st.selectbox("Fasting BS", CATEGORICAL_ALLOWED["fastingbs"])
            restingecg = st.selectbox("Resting ECG", CATEGORICAL_ALLOWED["restingecg"])
            maxhr = st.number_input("Max HR", int(DOMAIN_CONSTRAINTS["maxhr"][0]), int(DOMAIN_CONSTRAINTS["maxhr"][1]), value=150)
            exerciseangina = st.selectbox("Exercise Angina", CATEGORICAL_ALLOWED["exerciseangina"])
            oldpeak = st.number_input("Oldpeak", float(DOMAIN_CONSTRAINTS["oldpeak"][0]), float(DOMAIN_CONSTRAINTS["oldpeak"][1]), value=0.0, format="%.2f")

        with c3:
            st_slope = st.selectbox("ST Slope", CATEGORICAL_ALLOWED["st_slope"])
            st.write("")
            submit = st.form_submit_button("Predict ❤️", use_container_width=True)

        if submit:
            input_df = pd.DataFrame([{
                "age": age,
                "sex": sex,
                "chestpaintype": chestpaintype,
                "restingbp": restingbp,
                "cholesterol": cholesterol,
                "fastingbs": fastingbs,
                "restingecg": restingecg,
                "maxhr": maxhr,
                "exerciseangina": exerciseangina,
                "oldpeak": oldpeak,
                "st_slope": st_slope,
            }])

            ok, errs = validate_dataframe(input_df, require_columns=False)
            if not ok:
                st.error("⚠️ Validation errors:\n" + "\n".join(errs))
                return

            with st.spinner("🔄 Running prediction..."):
                preds = pred_service.predict(input_df)

            pred = int(preds[0])
            label = human_label_from_pred(pred)

            prob_text = ""
            try:
                if hasattr(pred_service, "predict_proba"):
                    probs = pred_service.predict_proba(input_df)
                    p_pos = float(probs[0][1]) if probs.shape[1] > 1 else float(probs[0])
                    prob_text = f" ({p_pos:.2%} probability)"
            except Exception:
                pass

            if pred == 1:
                st.error(f"🩺 Prediction: {label}{prob_text}")
            else:
                st.success(f"🩺 Prediction: {label}{prob_text}")

            st.write("#### Input data")
            st.table(input_df)


# ---------------------------
# UI — Batch Prediction
# ---------------------------
def batch_prediction_ui(pred_service: HeartDiseaseModel):
    st.header("📦 Batch Prediction (Upload CSV)")
    st.write("Upload a CSV with the following columns:")
    st.write([
        "age", "sex", "chestpaintype", "restingbp", "cholesterol",
        "fastingbs", "restingecg", "maxhr", "exerciseangina",
        "oldpeak", "st_slope",
    ])

    left, right = st.columns([2, 1])
    with right:
        st.download_button(
            "⬇️ Download template CSV",
            data=template_csv_bytes(),
            file_name="template_heart_disease.csv",
            mime="text/csv",
        )
        st.info("ℹ️ Tip: include header row exactly as shown above.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("📥 Upload a CSV to get predictions.")
        return

    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Uploaded Data (Preview)")
    st.dataframe(df.head(50))

    ok, errors = validate_dataframe(df, require_columns=True)
    if not ok:
        st.error("⚠️ Validation failed:")
        for e in errors:
            st.write(f"- {e}")
        return

    with st.spinner("🔄 Generating predictions..."):
        preds = pred_service.predict(df)

    df["heartdisease_prediction"] = [human_label_from_pred(int(p)) for p in preds]

    try:
        if hasattr(pred_service, "predict_proba"):
            proba = pred_service.predict_proba(df)
            if proba.ndim == 2 and proba.shape[1] > 1:
                df["probability_heartdisease"] = proba[:, 1]
            else:
                df["probability_heartdisease"] = proba[:, 0]
    except Exception:
        pass

    st.success("🎉 Predictions generated!")
    st.write("### 🔍 Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")


# ---------------------------
# MAIN APP
# ---------------------------
def main():
    st.sidebar.title("❤️ Heart Disease Prediction")

    # ---------------------------
    # Training section
    # ---------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Model Status")

    if model_exists():
        st.sidebar.success("✔️ Model Loaded")
        allow_training = False
    else:
        st.sidebar.error("❌ Model Not Found")
        allow_training = True

    # Disable training inside Docker
    if RUNNING_IN_DOCKER:
        st.sidebar.warning("Training is disabled inside Docker runtime.")
    else:
        if st.sidebar.button("🧠 Train Model"):
            if allow_training:
                new_service = train_pipeline_and_reload()
                if new_service:
                    st.session_state["prediction_service"] = new_service
                    st.sidebar.success("✔️ Model Ready")
            else:
                st.sidebar.info("ℹ️ Model already exists. Delete files to retrain.")

    st.sidebar.markdown("---")

    # Load or reuse model
    if "prediction_service" not in st.session_state:
        try:
            st.session_state["prediction_service"] = load_prediction_service()
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load model: {e}")
            st.stop()

    prediction_service = st.session_state["prediction_service"]

    # ---------------------------
    # Main menu
    # ---------------------------
    menu = st.sidebar.radio("Menu", ["Single Prediction", "Batch Prediction", "About / Help"])

    if menu == "Single Prediction":
        single_prediction_ui(prediction_service)

    elif menu == "Batch Prediction":
        batch_prediction_ui(prediction_service)

    elif menu == "About / Help":
        st.title("ℹ️ About This App")
        st.markdown(
            """
            - Simple and clean Streamlit UI  
            - Uses preprocessor + model from the `final_model/` folder  
            - Supports **single & batch predictions**  
            - Validates inputs before prediction  
            - Training is fully disabled inside Docker  
            """
        )
        st.json({
            "categorical_allowed": CATEGORICAL_ALLOWED,
            "domain_constraints": DOMAIN_CONSTRAINTS,
        })


if __name__ == "__main__":
    main()
