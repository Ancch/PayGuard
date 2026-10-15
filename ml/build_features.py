import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Load synthetic data
DATA_PATH = os.path.join('data', 'synthetic_transactions.csv')
df = pd.read_csv(DATA_PATH)

# Encode categorical columns
cat_cols = ['payment_direction', 'expected_merchant', 'actual_beneficiary']
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
cat_features = enc.fit_transform(df[cat_cols])
cat_feature_names = enc.get_feature_names_out(cat_cols)

# Numerical features
num_cols = ['amount_inr', 'is_new_payee', 'payee_txn_count_30d', 'device_risk_signal']
num_features = df[num_cols].astype(float).values

# Combine features
X = np.hstack([num_features, cat_features])

y = df['fraud_label'].values

# To avoid leakage, split by sender_account groups
unique_accounts = df['sender_account'].unique()
train_accounts, temp_accounts = train_test_split(unique_accounts, test_size=0.3, random_state=42)
val_accounts, test_accounts = train_test_split(temp_accounts, test_size=0.5, random_state=42)

# Helper to filter rows by account sets
def filter_by_accounts(accounts):
    return df['sender_account'].isin(accounts)

train_idx = filter_by_accounts(train_accounts)
val_idx = filter_by_accounts(val_accounts)
test_idx = filter_by_accounts(test_accounts)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Save processed arrays
os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
np.save(os.path.join('data', 'processed', 'X_train.npy'), X_train)
np.save(os.path.join('data', 'processed', 'y_train.npy'), y_train)
np.save(os.path.join('data', 'processed', 'X_val.npy'), X_val)
np.save(os.path.join('data', 'processed', 'y_val.npy'), y_val)
np.save(os.path.join('data', 'processed', 'X_test.npy'), X_test)
np.save(os.path.join('data', 'processed', 'y_test.npy'), y_test)

# Persist encoders and scalers for later use
joblib.dump(enc, os.path.join('ml', 'models', 'onehot_encoder.joblib'))
print('Feature engineering completed and data saved.')
