from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.preprocessor import DataPreprocessor
    from src.feature_engine import FeatureEngine
    from src.predictor import FraudPredictor
    from src.continuous_learner import ModelUpdater
    
    logger.info("Initializing ML pipeline components...")
    preprocessor = DataPreprocessor()
    feature_engine = FeatureEngine()
    predictor = FraudPredictor()
    learner = ModelUpdater()
    logger.info("ML pipeline initialized successfully.")
    
except ImportError as e:
    logger.warning(f"Could not import ML pipeline modules: {e}. Running in stub mode.")
    preprocessor, feature_engine, predictor, learner = None, None, None, None


app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions using strict MVC-like architecture.",
    version="1.0.0"
)

class TransactionRequest(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float

class PredictionResponse(BaseModel):
    transaction_id: str
    prediction: int
    risk_probability: float

class BatchUpdateRequest(BaseModel):
    transactions: List[Dict[str, Any]]
    target_column: str = "isFraud"

class BatchUpdateResponse(BaseModel):
    status: str
    message: str
    processed_count: int


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Fraud Detection API is running."}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    """
    Predict if a transaction is fraudulent.
    Routes the transaction through Preprocessing -> Feature Engineering -> Prediction.
    """
    if not predictor:
        raise HTTPException(status_code=500, detail="ML pipeline not loaded.")
        
    try:
        tx_data = transaction.model_dump()
        df_raw = pd.DataFrame([tx_data])
        
        df_cleaned = preprocessor.clean_data(df_raw)
        
        df_features = feature_engine.prepare_features(df_cleaned)
        
        pred_class, probability = predictor.predict(df_features)
        
        return PredictionResponse(
            transaction_id=f"{transaction.nameOrig}-{transaction.nameDest}-{transaction.step}",
            prediction=int(pred_class),
            risk_probability=float(probability)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-model", response_model=BatchUpdateResponse)
def update_model(batch: BatchUpdateRequest):
    """
    Trigger continuous learning with a batch of new labeled transactions.
    Routes to continuous_learner to update the weights.
    """
    if not learner:
        raise HTTPException(status_code=500, detail="Continuous Learner module not loaded.")
        
    try:
        if not batch.transactions:
            raise HTTPException(status_code=400, detail="No transactions provided in the batch.")
            
        df_batch = pd.DataFrame(batch.transactions)
        
        if batch.target_column not in df_batch.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{batch.target_column}' missing from batch data.")
            
        y_new = df_batch[batch.target_column]
        X_new_raw = df_batch.drop(columns=[batch.target_column])
        
        X_new_cleaned = preprocessor.clean_data(X_new_raw)
        X_new_features = feature_engine.prepare_features(X_new_cleaned)
        
        update_success = learner.update_model_batch(X_new_features, y_new)
        
        if update_success:
            return BatchUpdateResponse(
                status="success",
                message="Model successfully updated with the new batch.",
                processed_count=len(df_batch)
            )
        else:
            raise HTTPException(status_code=500, detail="Model update failed internally.")
            
    except Exception as e:
        logger.error(f"Batch update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
