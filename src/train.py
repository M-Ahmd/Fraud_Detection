import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

from src.preprocessor import DataPreprocessor
from src.feature_engine import FeatureEngine

def train_initial_models():
    filename = "PS_20174392719_1491204439457_log.csv"
    
    df = pd.read_csv(filename) 
    
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.clean_data(df)
    
    y = df_cleaned['isFraud']
    X_raw = df_cleaned.drop(columns=['isFraud'])
    
    feature_engine = FeatureEngine()
    X_final = feature_engine.fit_and_prepare_features(X_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )
    
    weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    xgb_final = XGBClassifier(
        n_estimators=200,      
        learning_rate=0.1,     
        max_depth=6,           
        scale_pos_weight=weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_final.fit(X_train, y_train)
    
    y_pred = xgb_final.predict(X_test)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    disp.plot(cmap='Reds')
    plt.title('Final XGBoost Confusion Matrix (Production Script)')
    plt.show() 
    
    os.makedirs("models_saved", exist_ok=True)
    joblib.dump(xgb_final, "models_saved/xgboost_model.pkl")

if __name__ == "__main__":
    train_initial_models()