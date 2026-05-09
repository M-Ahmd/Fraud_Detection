import os
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class FeatureEngine:
    def __init__(self, scaler_model_path: str = "models_saved/scaler.pkl", pca_model_path: str = "models_saved/pca_model.pkl", kmeans_model_path: str = "models_saved/kmeans_model.pkl"):
        self.scaler_model_path = scaler_model_path
        self.pca_model_path = pca_model_path
        self.kmeans_model_path = kmeans_model_path
        
        try:
            self.scaler = joblib.load(self.scaler_model_path)
            self.pca = joblib.load(self.pca_model_path)
            self.kmeans = joblib.load(self.kmeans_model_path)
        except Exception:
            self.scaler = None
            self.pca = None
            self.kmeans = None

    def fit_and_prepare_features(self, df_cleaned: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the Scaler, PCA, and KMeans models, saves them to disk, 
        and returns the engineered DataFrame. Designed for training mode.
        """
        df_final = df_cleaned.copy()
        os.makedirs("models_saved", exist_ok=True)

        cols_to_scale = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
        existing_scale_cols = [c for c in cols_to_scale if c in df_final.columns]
        
        self.scaler = RobustScaler()
        if existing_scale_cols:
            df_final[existing_scale_cols] = self.scaler.fit_transform(df_final[existing_scale_cols])
        joblib.dump(self.scaler, self.scaler_model_path)


        all_cols = sorted(df_final.columns.tolist())
        df_all = df_final[all_cols]
        self.pca = PCA(n_components=3, random_state=42)
        pca_output = self.pca.fit_transform(df_all)
        joblib.dump(self.pca, self.pca_model_path)

        self.kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init='auto')
        cluster_preds = self.kmeans.fit_predict(pca_output)
        joblib.dump(self.kmeans, self.kmeans_model_path)

        df_final['behavioral_segment'] = cluster_preds
        final_cols = all_cols + ['behavioral_segment']
        return df_final[final_cols]

    def prepare_features(self, df_cleaned: pd.DataFrame) -> pd.DataFrame:
        """
        Applies previously fitted models to transform the data. Designed for inference mode.
        """
        if self.pca is None or self.kmeans is None or self.scaler is None:
            raise ValueError("Models not loaded properly for inference.")

        df_final = df_cleaned.copy()

        # 1. Scale continuous features
        cols_to_scale = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
        existing_scale_cols = [c for c in cols_to_scale if c in df_final.columns]
        
        if existing_scale_cols:
            df_final[existing_scale_cols] = self.scaler.transform(df_final[existing_scale_cols])

        numeric_cols = sorted(df_final.columns.tolist())
        df_numeric = df_final[numeric_cols]

        pca_output = self.pca.transform(df_numeric)

        cluster_preds = self.kmeans.predict(pca_output)

        df_final['behavioral_segment'] = cluster_preds
        final_cols = numeric_cols + ['behavioral_segment']
        return df_final[final_cols]
