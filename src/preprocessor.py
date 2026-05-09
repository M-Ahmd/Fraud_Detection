import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.type_categories = ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes raw data, drops 100% correlated features (e.g., isFlaggedFraud),
        handles encoding, and returns the cleaned DataFrame.
        """
        df_cleaned = df.copy()
        
        if 'step' in df_cleaned.columns:
            df_cleaned['hour_of_day'] = df_cleaned['step'] % 24
            
        if 'nameDest' in df_cleaned.columns:
            df_cleaned['DestType'] = df_cleaned['nameDest'].str[0]
            df_cleaned['DestType'] = df_cleaned['DestType'].map({'C': 0, 'M': 1}).fillna(0).astype(int)

        df_cleaned = df_cleaned.drop(columns=['isFlaggedFraud', 'nameOrig', 'nameDest'], errors='ignore')
        
        if 'type' in df_cleaned.columns:
            for cat in self.type_categories:
                col_name = f"type_{cat}"
                df_cleaned[col_name] = (df_cleaned['type'] == cat).astype(int)
            df_cleaned = df_cleaned.drop(columns=['type'])
            
        return df_cleaned
