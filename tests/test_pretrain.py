import os
import pandas as pd
import pytest

def test_data_file_exists():
    data_path = "data/prepared/train.csv"
    assert os.path.exists(data_path), f"Training data not found: {data_path}"

def test_data_schema():
    train_data = pd.read_csv("data/prepared/train.csv")
    test_data = pd.read_csv("data/prepared/test.csv")
    
    required_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
    
    missing_in_train = set(required_columns) - set(train_data.columns)
    missing_in_test = set(required_columns) - set(test_data.columns)
    
    assert not missing_in_train, f"Missing columns in train: {missing_in_train}"
    assert not missing_in_test, f"Missing columns in test: {missing_in_test}"

def test_data_no_critical_missing():
    train_data = pd.read_csv("data/prepared/train.csv")
    test_data = pd.read_csv("data/prepared/test.csv")
    
    assert train_data.isnull().sum().sum() == 0, "Train data has missing values"
    assert test_data.isnull().sum().sum() == 0, "Test data has missing values"

def test_target_variable_distribution():
    train_data = pd.read_csv("data/prepared/train.csv")
    
    assert 'stroke' in train_data.columns, "Target column 'stroke' not found"
    assert set(train_data['stroke'].unique()).issubset({0, 1}), "Target has invalid values"
    
    positive_ratio = train_data['stroke'].sum() / len(train_data)
    assert 0.01 < positive_ratio < 0.5, f"Unexpected target distribution: {positive_ratio:.2%}"

def test_minimum_samples():
    train_data = pd.read_csv("data/prepared/train.csv")
    
    assert len(train_data) >= 1000, f"Too few training samples: {len(train_data)}"
    
    positive_samples = train_data['stroke'].sum()
    assert positive_samples >= 50, f"Too few positive samples: {positive_samples}"

def test_feature_value_ranges():
    train_data = pd.read_csv("data/prepared/train.csv")
    
    assert train_data['age'].min() >= 0 and train_data['age'].max() <= 120, "Age out of reasonable range"
    assert train_data['avg_glucose_level'].min() >= 50, "Glucose level unreasonably low"
