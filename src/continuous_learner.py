import pandas as pd
import joblib
import xgboost as xgb

class ModelUpdater:
    def __init__(self, xgboost_model_path: str = "models_saved/xgboost_model.pkl"):
        self.xgboost_model_path = xgboost_model_path

    def update_model_batch(self, X_new: pd.DataFrame, y_new: pd.Series):
        """
        Loads the existing xgboost_model.pkl, calculates the new scale_pos_weight 
        based on the new batch, and trains using xgb_model=existing_model to perform 
        true partial/batch training without historical data. 
        Saves the updated .pkl back to the folder.
        """
        try:
            existing_model = joblib.load(self.xgboost_model_path)

            num_negative = (y_new == 0).sum()
            num_positive = (y_new == 1).sum()
            
            new_scale_pos_weight = num_negative / num_positive if num_positive > 0 else 1.0

            existing_model.set_params(scale_pos_weight=new_scale_pos_weight)
            existing_model.fit(X_new, y_new, xgb_model=existing_model.get_booster())
            updated_model = existing_model

            joblib.dump(updated_model, self.xgboost_model_path)
            
            return True
            
        except Exception as e:
            print(f"Error during continuous learning: {e}")
            return False
