import pandas as pd
import numpy as np

def generate_api_test_batch(filename="new_batch_api_test.csv", num_rows=1000):
    
    np.random.seed(42)
    
    types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    
    data = {
        'step': np.random.randint(744, 800, num_rows), # خطوات في المستقبل
        'type': np.random.choice(types, num_rows, p=[0.2, 0.3, 0.1, 0.3, 0.1]),
        'amount': np.round(np.random.uniform(10, 100000, num_rows), 2),
        'nameOrig': ['C' + str(np.random.randint(10000000, 99999999)) for _ in range(num_rows)],
        'oldbalanceOrg': np.round(np.random.uniform(1000, 500000, num_rows), 2),
        'nameDest': ['C' + str(np.random.randint(10000000, 99999999)) for _ in range(num_rows)],
        'oldbalanceDest': np.round(np.random.uniform(0, 1000000, num_rows), 2),
        'isFlaggedFraud': 0
    }
    
    df = pd.DataFrame(data)
    
    df['newbalanceOrig'] = np.where(df['oldbalanceOrg'] >= df['amount'], 
                                    df['oldbalanceOrg'] - df['amount'], 0)
    df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']
    
    df['isFraud'] = 0
    fraud_indices = df.sample(frac=0.5, random_state=42).index
    df.loc[fraud_indices, 'isFraud'] = 1
    
    cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 
            'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']
    df = df[cols]
    
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    generate_api_test_batch()