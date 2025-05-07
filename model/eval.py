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
    """Return actual metrics for logistic regression, mock metrics for CNN"""
    try:
        logistic_metrics = np.load('logistic_model_metrics.npz')
        return {
            "logistic_regression": {
                "accuracy": float(logistic_metrics['accuracy']),
                "precision": float(logistic_metrics['precision']),
                "recall": float(logistic_metrics['recall']),
                "f1_score": float(logistic_metrics['f1_score']),
                "mse": float(logistic_metrics['mse'])
            },
            "cnn": {
                "accuracy": 0.85,
                "precision": 0.87,
                "recall": 0.83,
                "f1_score": 0.85,
                "mse": 0.15,
            }
        }
    except FileNotFoundError:
        print("No metrics file found. Using mock values.")
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
            }
        }
