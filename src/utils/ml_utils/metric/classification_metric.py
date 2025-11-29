from src.exception import HeartDiseaseException
from sklearn.metrics import classification_report
import sys

def get_classification_score(y_true, y_pred):
    try:
        return classification_report(y_true, y_pred)
    except Exception as e:
        raise HeartDiseaseException(e, sys)