import os
import sys
import pandas as pd

from src.exception import HeartDiseaseException

class HeartDiseaseModel:
    def __init__(self, preprocessor, model):

        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise HeartDiseaseException(e, sys)
    
    def feature_engineering(self, x: pd.DataFrame) -> pd.DataFrame:
        try:
            # --- oldpeak binning ---
            bins = [0, 1.0, 2.0, 4.0, float("inf")]
            labels = ["Normal", "Mild", "Moderate", "Severe"]
            x['oldpeak'] = pd.cut(x['oldpeak'], bins=bins, labels=labels, include_lowest=True, right=False)
            return x
        except Exception as e:
            raise HeartDiseaseException(e, sys)
    
    def predict(self, x):
        try:
            x_transformed = self.feature_engineering(x)
            x_transformed = self.preprocessor.transform(x)
            y_pred = self.model.predict(x_transformed)
            return y_pred
        except Exception as e:
            raise HeartDiseaseException(e, sys)