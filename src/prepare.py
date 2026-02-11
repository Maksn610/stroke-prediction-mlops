import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import os
import json

input_file = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)

print(f"Завантажено {df.shape[0]} рядків")

df_clean = df.dropna()
print(f"Після видалення пропусків: {df_clean.shape[0]} рядків")

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

X = df_clean.drop('stroke', axis=1)
y = df_clean['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_data = X_train.copy()
train_data['stroke'] = y_train
test_data = X_test.copy()
test_data['stroke'] = y_test

train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

stats = {
    "train_samples": len(train_data),
    "test_samples": len(test_data),
    "positive_class_train": int(y_train.sum()),
    "positive_class_test": int(y_test.sum())
}

with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print(f"Тренувальна вибірка: {len(train_data)}")
print(f"Тестова вибірка: {len(test_data)}")
print(f"Статистика збережена у stats.json")
