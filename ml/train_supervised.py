import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load processed data
X_train = np.load(os.path.join('data', 'processed', 'X_train.npy'))
y_train = np.load(os.path.join('data', 'processed', 'y_train.npy'))
X_val = np.load(os.path.join('data', 'processed', 'X_val.npy'))
y_val = np.load(os.path.join('data', 'processed', 'y_val.npy'))

# Hyperparameters (can be tuned later)
rf_params = {
    'n_estimators': 200,
    'max_depth': None,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
}

clf = RandomForestClassifier(**rf_params)
clf.fit(X_train, y_train)

# Validation evaluation
val_pred = clf.predict(X_val)
val_proba = clf.predict_proba(X_val)[:, 1]
print('Validation Classification Report')
print(classification_report(y_val, val_pred, digits=4))
print('Validation ROC AUC:', roc_auc_score(y_val, val_proba))

# Save model
model_dir = os.path.join('ml', 'models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'supervised_rf.joblib')
joblib.dump(clf, model_path)
print('Supervised RandomForest model saved to', model_path)
