from typing import Dict


def get_performance_metrics() -> Dict[str, Dict[str, float]]:
    """Return mock model performance metrics"""
    return {
        "logistic_regression": {
            "accuracy": 0.75,
            "precision": 0.78,
            "recall": 0.72,
            "f1_score": 0.75,
            "mse": 0.25,
        },
        "cnn": {
            "accuracy": 0.85,
            "precision": 0.87,
            "recall": 0.83,
            "f1_score": 0.85,
            "mse": 0.15,
        },
    }
