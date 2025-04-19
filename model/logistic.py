import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Union
import pandas as pd


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, reg_lambda: float = 0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.reg_lambda = reg_lambda  # L2 regularization parameter
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None

    def _get_feature_columns(self, X: pd.DataFrame) -> List[str]:
        """Get relevant feature columns for training/prediction"""
        return [col for col in X.columns if col not in ['path', 'label']]

    def initialize_parameters(self, n_features: int) -> None:
        """Initialize weights with small random values"""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the cost function with L2 regularization"""
        if self.weights is None:
            raise ValueError("Weights not initialized")
            
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        
        # Add epsilon to avoid log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Compute cross-entropy loss
        cost = (-1 / m) * np.sum(
            y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
        )
        
        # Add L2 regularization
        reg_cost = (self.reg_lambda / (2 * m)) * np.sum(self.weights ** 2)
        
        return float(cost + reg_cost)

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'LogisticRegression':
        """
        Fit the logistic regression model.
        X: DataFrame with features
        y: Series with labels (0 for Human, 1 for AI)
        """
        # Select and preprocess features
        self.feature_names = self._get_feature_columns(X)
        X_features = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Convert to numpy arrays
        X_scaled = np.array(X_scaled, dtype=np.float64)
        y_array = np.array(y, dtype=np.float64)
        
        # Get dimensions
        m, n = X_scaled.shape
        
        # Initialize parameters
        self.initialize_parameters(n)
        
        # Initialize best parameters
        best_cost = float('inf')
        best_weights = None
        best_bias = 0.0
        
        # Gradient descent with early stopping
        patience = 5
        min_improvement = 1e-4
        no_improvement = 0
        
        for i in range(self.num_iterations):
            if self.weights is None:
                raise ValueError("Weights not initialized")
                
            # Forward propagation
            z = np.dot(X_scaled, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Compute gradients
            dw = (1 / m) * np.dot(X_scaled.T, (predictions - y_array))
            dw += (self.reg_lambda / m) * self.weights
            db = (1 / m) * np.sum(predictions - y_array)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost and check for improvement
            current_cost = self.compute_cost(X_scaled, y_array)
            
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {current_cost:.6f}")
            
            if current_cost < best_cost - min_improvement:
                best_cost = current_cost
                best_weights = self.weights.copy()
                best_bias = self.bias
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Early stopping
            if no_improvement >= patience:
                print(f"Early stopping at iteration {i}")
                break
        
        # Use best parameters found
        self.weights = best_weights
        self.bias = best_bias
        
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of being AI-generated"""
        if self.feature_names is None or self.weights is None:
            raise ValueError("Model has not been fitted yet!")
            
        # Select and preprocess features
        X_features = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        # Compute probability
        z = np.dot(X_scaled, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict class (0 for Human, 1 for AI)"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores"""
        if self.feature_names is None or self.weights is None:
            raise ValueError("Model has not been fitted yet!")
            
        # Get absolute values of weights as importance scores
        importance = np.abs(self.weights)
        
        # Create dictionary of feature names and their importance scores
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
