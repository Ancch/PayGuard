import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             auc)

# Paths
DATA_DIR = os.path.join('data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR = os.path.join('reports')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Load test data
X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

# Helper to evaluate a model and save figures
def evaluate_model(name, model_path, scaler_path=None, need_scaling=False):
    model = joblib.load(model_path)
    if need_scaling and scaler_path:
        scaler = joblib.load(scaler_path)
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test
    y_pred = model.predict(X_eval)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_eval)[:, 1]
    else:
        # IsolationForest uses decision_function (higher -> normal), we invert
        if hasattr(model, 'decision_function'):
            y_score = -model.decision_function(X_eval)
        else:
            y_score = y_pred  # fallback
    # Metrics
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall, precision)

    # Save confusion matrix heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = os.path.join(FIG_DIR, f'{name.lower().replace(" ", "_")}_confusion.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # Save ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title(f'{name} - ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    roc_path = os.path.join(FIG_DIR, f'{name.lower().replace(" ", "_")}_roc.png')
    plt.savefig(roc_path)
    plt.close()

    # Save PR curve
    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
    plt.title(f'{name} - Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    pr_path = os.path.join(FIG_DIR, f'{name.lower().replace(" ", "_")}_pr.png')
    plt.savefig(pr_path)
    plt.close()

    return {
        'name': name,
        'report': report,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_path': cm_path,
        'roc_path': roc_path,
        'pr_path': pr_path,
    }

# Evaluate each model
results = []
# Supervised RandomForest
results.append(evaluate_model(
    name='RandomForest Classifier',
    model_path=os.path.join('ml', 'models', 'supervised_rf.joblib')
))
# IsolationForest anomaly detector
results.append(evaluate_model(
    name='IsolationForest Anomaly',
    model_path=os.path.join('ml', 'models', 'isolation_forest.joblib')
))
# Graph‑based LogisticRegression (needs scaling)
results.append(evaluate_model(
    name='Graph LogisticRegression',
    model_path=os.path.join('ml', 'models', 'graph_logreg.joblib'),
    scaler_path=os.path.join('ml', 'models', 'graph_scaler.joblib'),
    need_scaling=True
))

# Write markdown summary report
report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.md')
with open(report_path, 'w') as f:
    f.write('# Model Evaluation Report\n\n')
    for res in results:
        f.write(f"## {res['name']}\n\n")
        f.write('### Classification Report\n')
        f.write('```
')
        f.write(res['report'])
        f.write('\n```
\n')
        f.write(f"- ROC AUC: {res['roc_auc']:.4f}\n")
        f.write(f"- PR AUC: {res['pr_auc']:.4f}\n\n")
        f.write(f"![Confusion Matrix]({res['confusion_path']})\n\n")
        f.write(f"![ROC Curve]({res['roc_path']})\n\n")
        f.write(f"![PR Curve]({res['pr_path']})\n\n")

print('Evaluation report generated at', report_path)
