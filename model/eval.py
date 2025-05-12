from typing import Dict
import numpy as np


# Given prdiction reults in list of 0 and 1, and


def calculate_accuracy(y_pred, y_true):
    """Calculate accuracy for a given model and test data loader"""
    y_pred_binary = (y_pred >= 0.5).astype(int)
    return np.mean(y_pred_binary == y_true)


def calculate_precision(y_pred, y_true):
    """Calculate precision for a given model and test data loader"""
    y_pred_binary = (y_pred >= 0.5).astype(int)
    true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred_binary == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0


def calculate_recall(y_pred, y_true):
    """Calculate recall for a given model and test data loader"""
    y_pred_binary = (y_pred >= 0.5).astype(int)
    true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0.0


def calculate_f1_score(y_pred, y_true):
    """Calculate F1 score for a given model and test data loader"""
    precision = calculate_precision(y_pred, y_true)
    recall = calculate_recall(y_pred, y_true)
    return (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


def calculate_mse(y_pred, y_true):
    """Calculate Mean Squared Error"""
    return np.mean((y_pred - y_true) ** 2)


def get_performance_metrics(y_pred, y_true) -> Dict[str, float]:
    """Calculate and return performance metrics for predictions

    Args:
        y_pred: Model predictions (probabilities between 0-1)
        y_true: True labels (0 or 1)

    Returns:
        Dictionary containing various performance metrics
    """
    return {
        "accuracy": float(calculate_accuracy(y_pred > 0.5, y_true)),
        "precision": float(calculate_precision(y_pred > 0.5, y_true)),
        "recall": float(calculate_recall(y_pred > 0.5, y_true)),
        "f1_score": float(calculate_f1_score(y_pred > 0.5, y_true)),
        "mse": float(calculate_mse(y_pred, y_true)),
    }
