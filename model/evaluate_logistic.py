import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

def plot_probability_distribution(y_test, y_pred_proba):
    """Plot the distribution of prediction probabilities"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pd.DataFrame({
        'Probability': y_pred_proba,
        'True Label': ['AI' if y else 'Human' for y in y_test]
    }), x='Probability', hue='True Label', bins=30)
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Probability of being AI-generated')
    plt.ylabel('Count')
    plt.show()

def plot_feature_importance(model):
    """Plot feature importance scores"""
    importance = model.get_feature_importance()
    plt.figure(figsize=(12, 6))
    features = list(importance.keys())
    scores = list(importance.values())
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

def plot_roc_pr_curves(y_test, y_pred_proba):
    """Plot ROC and Precision-Recall curves"""
    # Calculate curves
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(12, 5))

    # Plot ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

def print_threshold_analysis(y_test, y_pred_proba, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Print performance metrics at different probability thresholds"""
    print("\nPerformance at different thresholds:")
    print("=" * 50)
    print(f"{'Threshold':^10} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1':^10}")
    print("-" * 50)

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{threshold:^10.2f} | {acc:^10.3f} | {prec:^10.3f} | {rec:^10.3f} | {f1:^10.3f}")

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation"""
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)
    
    print("\nModel Performance Summary:")
    print("=" * 50)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    # Generate all plots
    plot_probability_distribution(y_test, y_pred_proba)
    plot_feature_importance(model)
    plot_roc_pr_curves(y_test, y_pred_proba)
    print_threshold_analysis(y_test, y_pred_proba)

    