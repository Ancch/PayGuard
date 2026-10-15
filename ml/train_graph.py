import os
import pandas as pd
import numpy as np
import networkx as nx
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load raw data
DATA_PATH = os.path.join('data', 'synthetic_transactions.csv')
df = pd.read_csv(DATA_PATH)

# Build directed graph of all transactions (sender -> receiver)
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['sender_account'], row['receiver_account'], amount=row['amount_inr'], timestamp=row['timestamp'])

# Compute per-node graph features
node_features = {}
for node in G.nodes:
    in_deg = G.in_degree(node)
    out_deg = G.out_degree(node)
    total_deg = in_deg + out_deg
    # Average transaction amount of outgoing edges (if any)
    out_edges = G.out_edges(node, data=True)
    if out_deg > 0:
        avg_out_amount = np.mean([d['amount'] for _, _, d in out_edges])
    else:
        avg_out_amount = 0.0
    node_features[node] = {
        'in_degree': in_deg,
        'out_degree': out_deg,
        'total_degree': total_deg,
        'avg_out_amount': avg_out_amount,
    }

# Map node features back to transactions (use sender's features)
sender_feats = []
receiver_feats = []
for _, row in df.iterrows():
    s_feat = node_features.get(row['sender_account'], {'in_degree':0,'out_degree':0,'total_degree':0,'avg_out_amount':0})
    r_feat = node_features.get(row['receiver_account'], {'in_degree':0,'out_degree':0,'total_degree':0,'avg_out_amount':0})
    sender_feats.append([s_feat['in_degree'], s_feat['out_degree'], s_feat['total_degree'], s_feat['avg_out_amount']])
    receiver_feats.append([r_feat['in_degree'], r_feat['out_degree'], r_feat['total_degree'], r_feat['avg_out_amount']])

sender_feats = np.array(sender_feats)
receiver_feats = np.array(receiver_feats)

# Load previously engineered numeric + one‑hot features (to keep consistency)
X_num = np.load(os.path.join('data', 'processed', 'X_train.npy'))  # placeholder, will be replaced
# Instead, recreate the same base features used in build_features.py
cat_cols = ['payment_direction', 'expected_merchant', 'actual_beneficiary']
enc = joblib.load(os.path.join('ml', 'models', 'onehot_encoder.joblib'))
cat_features = enc.transform(df[cat_cols])
num_cols = ['amount_inr', 'is_new_payee', 'payee_txn_count_30d', 'device_risk_signal']
num_features = df[num_cols].astype(float).values
base_X = np.hstack([num_features, cat_features])

# Combine base features with graph features (sender + receiver)
X = np.hstack([base_X, sender_feats, receiver_feats])

y = df['fraud_label'].values

# Split by sender_account groups (same as in build_features.py)
unique_accounts = df['sender_account'].unique()
from sklearn.model_selection import train_test_split
train_accounts, temp_accounts = train_test_split(unique_accounts, test_size=0.3, random_state=42)
val_accounts, test_accounts = train_test_split(temp_accounts, test_size=0.5, random_state=42)

def mask_by_accounts(accounts):
    return df['sender_account'].isin(accounts)

train_mask = mask_by_accounts(train_accounts)
val_mask = mask_by_accounts(val_accounts)
test_mask = mask_by_accounts(test_accounts)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Logistic Regression model (simple, needs scaling)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
logreg.fit(X_train_scaled, y_train)

# Validation metrics
val_pred = logreg.predict(X_val_scaled)
val_proba = logreg.predict_proba(X_val_scaled)[:, 1]
print('Graph‑based LogisticRegression validation report:')
print(classification_report(y_val, val_pred, digits=4))
print('ROC AUC:', roc_auc_score(y_val, val_proba))

# Save scaler and model
model_dir = os.path.join('ml', 'models')
os.makedirs(model_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(model_dir, 'graph_scaler.joblib'))
joblib.dump(logreg, os.path.join(model_dir, 'graph_logreg.joblib'))
print('Graph model and scaler saved.')
