import pandas as pd
import joblib
import xgboost as xgb

class FraudPredictor:
    def __init__(self, xgboost_model_path: str = "models_saved/xgboost_model.pkl"):
        self.xgboost_model_path = xgboost_model_path
        try:
            self.model = joblib.load(xgboost_model_path)
        except Exception as e:
            self.model = None

    def reload_model(self):
        """Reload the model from disk after continuous learning updates."""
        self.model = joblib.load(self.xgboost_model_path)

    def predict(self, df_features: pd.DataFrame, threshold: float = 0.5):
        """
        Runs inference using XGBClassifier and returns prediction and risk probability.
        """
        if self.model is None:
            raise ValueError("XGBoost model not loaded properly.")

        probs = self.model.predict_proba(df_features)[:, 1]
        
        preds = (probs > threshold).astype(int)
        
        return preds[0], probs[0]