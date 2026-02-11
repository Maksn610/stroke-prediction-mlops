import os
import json
import pytest

def test_model_artifact_exists():
    assert os.path.exists("models/best_optimized_model.pkl"), "Model file not found"

def test_metrics_file_exists():
    assert os.path.exists("data/models/metrics.json"), "Metrics file not found"

def test_metrics_format_valid():
    with open("data/models/metrics.json", "r") as f:
        metrics = json.load(f)
    
    required_metrics = {"test_accuracy", "test_f1", "test_roc_auc"}
    missing = required_metrics - set(metrics.keys())
    assert not missing, f"Missing metrics: {missing}"

def test_quality_gate_f1():
    f1_threshold = float(os.getenv("F1_THRESHOLD", "0.15"))
    
    with open("data/models/metrics.json", "r") as f:
        metrics = json.load(f)
    
    f1_score = float(metrics["test_f1"])
    assert f1_score >= f1_threshold, (
        f"Quality Gate FAILED: F1-Score {f1_score:.4f} < threshold {f1_threshold:.4f}"
    )

def test_quality_gate_roc_auc():
    roc_threshold = float(os.getenv("ROC_THRESHOLD", "0.70"))
    
    with open("data/models/metrics.json", "r") as f:
        metrics = json.load(f)
    
    roc_auc = float(metrics["test_roc_auc"])
    assert roc_auc >= roc_threshold, (
        f"ROC-AUC below threshold: {roc_auc:.4f} < {roc_threshold:.4f}"
    )

def test_no_overfitting():
    with open("data/models/metrics.json", "r") as f:
        metrics = json.load(f)
    
    overfitting_gap = metrics["train_f1"] - metrics["test_f1"]
    max_acceptable_gap = 0.3
    
    assert overfitting_gap <= max_acceptable_gap, (
        f"Excessive overfitting: gap={overfitting_gap:.4f} > {max_acceptable_gap}"
    )
