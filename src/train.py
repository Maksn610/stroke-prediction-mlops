import pandas as pd
import json
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

input_dir = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

train_data = pd.read_csv(os.path.join(input_dir, 'train.csv'))
test_data = pd.read_csv(os.path.join(input_dir, 'test.csv'))

X_train = train_data.drop('stroke', axis=1)
y_train = train_data['stroke']

X_test = test_data.drop('stroke', axis=1)
y_test = test_data['stroke']

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("Тренування моделі...")
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
y_pred_proba_test = model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)

y_pred_train = model.predict(X_train)
train_f1 = f1_score(y_train, y_pred_train)

metrics = {
    "test_accuracy": float(test_accuracy),
    "test_f1": float(test_f1),
    "test_roc_auc": float(test_roc_auc),
    "train_f1": float(train_f1),
    "overfitting_gap": float(train_f1 - test_f1)
}

# Збережи модель у data/models/
model_path = os.path.join(output_dir, 'model.pkl')
joblib.dump(model, model_path)

# ДОДАЙ: Збережи модель також у models/ для CI/CD
os.makedirs('models', exist_ok=True)
model_ci_path = 'models/best_optimized_model.pkl'
joblib.dump(model, model_ci_path)

metrics_path = os.path.join(output_dir, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Модель збережена: {model_path}")
print(f"Модель для CI: {model_ci_path}")
print(f"Метрики збережені: {metrics_path}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Test ROC-AUC: {test_roc_auc:.4f}")
