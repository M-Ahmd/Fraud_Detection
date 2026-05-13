import pandas as pd
import numpy as np

def generate_api_test_batch(filename="new_batch_api_test.csv", num_rows=10000):
    
    np.random.seed(42)
    
    data = {
        'step': np.random.randint(744, 800, num_rows),
        'type': ['PAYMENT'] * num_rows,  # All PAYMENT transactions
        'amount': np.round(np.random.uniform(10, 100000, num_rows), 2),
        'nameOrig': ['C' + str(np.random.randint(10000000, 99999999)) for _ in range(num_rows)],
        'oldbalanceOrg': np.round(np.random.uniform(1000, 500000, num_rows), 2),
        'nameDest': ['M' + str(np.random.randint(10000000, 99999999)) for _ in range(num_rows)],
        'oldbalanceDest': np.round(np.random.uniform(0, 1000000, num_rows), 2),
        'isFlaggedFraud': 0
    }
    
    df = pd.DataFrame(data)
    
    # Default: legitimate transactions (partial withdrawal)
    df['newbalanceOrig'] = np.where(df['oldbalanceOrg'] >= df['amount'], 
                                    df['oldbalanceOrg'] - df['amount'], 0)
    df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']
    df['isFraud'] = 0
    
    # Fraud pattern: account gets fully drained (amount >= oldbalanceOrg, newbalanceOrig = 0)
    # Use 80% fraud rate so the new trees learn a strong fraud signal
    fraud_indices = df.sample(frac=0.8, random_state=42).index
    df.loc[fraud_indices, 'amount'] = df.loc[fraud_indices, 'oldbalanceOrg'] + np.random.uniform(0, 100, len(fraud_indices))
    df.loc[fraud_indices, 'newbalanceOrig'] = 0.0
    df.loc[fraud_indices, 'newbalanceDest'] = df.loc[fraud_indices, 'oldbalanceDest'] + df.loc[fraud_indices, 'amount']
    df.loc[fraud_indices, 'isFraud'] = 1
    
    cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 
            'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']
    df = df[cols]
    
    df.to_csv(filename, index=False)
    print(f"Generated {num_rows} PAYMENT transactions ({len(fraud_indices)} fraud with drain pattern)")

if __name__ == "__main__":
    generate_api_test_batch()