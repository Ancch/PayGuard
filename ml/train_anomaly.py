import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load processed data (only legitimate records for training)
X_train = np.load(os.path.join('data', 'processed', 'X_train.npy'))
y_train = np.load(os.path.join('data', 'processed', 'y_train.npy'))

# Use only legitimate (label 0) for training the anomaly detector
X_train_legit = X_train[y_train == 0]

# Hyperparameters (can be adjusted)
iso_params = {
    'n_estimators': 200,
    'contamination': 'auto',
    'random_state': 42,
    'behaviour': 'new',
}

clf = IsolationForest(**iso_params)
clf.fit(X_train_legit)

# Validation on the whole validation set
X_val = np.load(os.path.join('data', 'processed', 'X_val.npy'))
y_val = np.load(os.path.join('data', 'processed', 'y_val.npy'))

# IsolationForest predicts -1 for outliers, 1 for inliers
val_pred_raw = clf.predict(X_val)
val_pred = (val_pred_raw == -1).astype(int)  # map outliers to fraud label 1
val_scores = -clf.decision_function(X_val)  # higher = more anomalous

print('Validation Classification Report (Anomaly Detector)')
print(classification_report(y_val, val_pred, digits=4))
print('Validation ROC AUC:', roc_auc_score(y_val, val_scores))

# Save model
model_dir = os.path.join('ml', 'models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'isolation_forest.joblib')
joblib.dump(clf, model_path)
print('IsolationForest model saved to', model_path)
