import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Union
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing.analyze import load_or_process_dataset


class LogisticRegression:
    # Define the top features as a class constant
    TOP_FEATURES = [
        'entropy',
        'line_count',
        'saturation_mean',
        'saturation_quartile_3',
        'red_mean',
        'hue_bin_0',
        'saturation_median',
        'local_contrast_3x3_std',
        'saturation_quartile_1',
        'hue_bin_3'
    ]

    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, reg_lambda: float = 0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.reg_lambda = reg_lambda  # L2 regularization parameter
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.scaler = StandardScaler()
        self.feature_names = self.TOP_FEATURES

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
            self.bias = float(self.bias - self.learning_rate * db)  # Explicitly cast to float
            
            # Compute cost and check for improvement
            current_cost = self.compute_cost(X_scaled, y_array)
            
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {current_cost:.6f}")
            
            if current_cost < best_cost - min_improvement:
                best_cost = current_cost
                best_weights = np.array(self.weights) if self.weights is not None else None
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of being AI-generated"""
        if self.weights is None:
            raise ValueError("Model has not been fitted yet!")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Compute probability
        z = np.dot(X_scaled, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
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

    def save_weights(self, path: str) -> None:
        """Save model weights and parameters to file"""
        if self.weights is None:
            raise ValueError("Model has not been fitted yet!")
        
        np.savez(
            path,
            weights=self.weights,
            bias=self.bias,
            scaler_mean=self.scaler.mean_,
            scaler_scale=self.scaler.scale_
        )

    def load_weights(self, path: str) -> None:
        """Load model weights and parameters from file"""
        data = np.load(path)
        
        self.weights = data['weights']
        self.bias = float(data['bias'])
        
        # Reconstruct the scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = data['scaler_mean']
        self.scaler.scale_ = data['scaler_scale']
        self.scaler.n_features_in_ = len(self.TOP_FEATURES)

def train_and_evaluate_model(df: pd.DataFrame):
    """Train and evaluate the logistic regression model"""
    # Select only the top features
    X = df[LogisticRegression.TOP_FEATURES]
    y = (df['label'] == 'AI').astype(int)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save the trained model weights
    model.save_weights('logistic_model_weights.npz')
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Print evaluation metrics
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    return model, (X_train, X_test, y_train, y_test)

def predict_single_image(image_array: np.ndarray) -> Dict[str, Union[float, str]]:
    """Predict whether a single image is AI-generated"""
    from data_preprocessing.analyze import (
        extract_color_features,
        extract_texture_features,
        extract_line_features,
        extract_contrast_features
    )
    
    # Extract all features
    features = {
        **extract_color_features(image_array),
        **extract_texture_features(image_array),
        **extract_line_features(image_array),
        **extract_contrast_features(image_array)
    }
    
    # Select only the top features
    selected_features = {f: features[f] for f in LogisticRegression.TOP_FEATURES}
    X = pd.DataFrame([selected_features])
    
    # Load model and predict
    model = LogisticRegression()
    try:
        model.load_weights('logistic_model_weights.npz')
        probability = model.predict_proba(X)[0]
        prediction = 'AI' if probability >= 0.5 else 'Human'
        
        return {
            'probability': float(probability),
            'prediction': prediction
        }
    except FileNotFoundError:
        raise ValueError("No trained model weights found. Please train the model first.")

if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    df = load_or_process_dataset()
    
    # Train and evaluate the model
    model, (X_train, X_test, y_train, y_test) = train_and_evaluate_model(df)
    
    # Example of using the model for prediction
    # Note: Replace with an actual image path
    # result = predict_single_image(model, "path/to/your/image.jpg")
    # print(f"\nPrediction for single image:")
    # print(f"Class: {result['prediction']}")
    # print(f"Probability: {result['probability']:.4f}")
