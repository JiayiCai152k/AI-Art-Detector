from typing import Dict
import numpy as np


# Given prdiction reults in list of 0 and 1, and


def calculate_accuracy(y_pred, y_true):
    """Calculate accuracy for a given model and test data loader"""
    return np.sum(y_pred.flatten() == y_true.flatten()) / len(y_true.flatten())


def calculate_precision(y_pred, y_true):
    """Calculate precision for a given model and test data loader"""
    return np.sum(y_pred.flatten() == y_true.flatten()) / len(y_true.flatten())


def calculate_recall(y_pred, y_true):
    """Calculate recall for a given model and test data loader"""
    return np.sum(y_pred.flatten() == y_true.flatten()) / len(y_true.flatten())


def calculate_f1_score(y_pred, y_true):
    """Calculate F1 score for a given model and test data loader"""
    return (
        2
        * (calculate_precision(y_pred, y_true) * calculate_recall(y_pred, y_true))
        / (calculate_precision(y_pred, y_true) + calculate_recall(y_pred, y_true))
    )


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
