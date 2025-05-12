import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Union
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error
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

    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 2000, reg_lambda: float = 0.01):
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
        # Use clipping for numerical stability
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def compute_cost(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> float:
        """Compute the cost function with L2 regularization"""
        if self.weights is None:
            raise ValueError("Weights not initialized")
            
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            X_features = X
        else:
            X_features = pd.DataFrame(X, columns=self.feature_names)
        
        m = X_features.shape[0]
        # Get predicted probabilities
        predictions = self.predict_proba(X_features)
        
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
        patience = 20
        min_improvement = 1e-4
        no_improvement = 0
        
        for i in range(self.num_iterations):
            if self.weights is None:
                raise ValueError("Weights not initialized")
                
            # Forward propagation
            predictions = self.predict_proba(X_features)
            
            # Compute gradients (simplified calculation)
            dw = (1 / m) * np.dot(X_scaled.T, (predictions - y_array))
            dw += (self.reg_lambda / m) * self.weights
            db = (1 / m) * np.sum(predictions - y_array)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias = float(self.bias - self.learning_rate * db)
            
            # Compute cost and check for improvement
            current_cost = self.compute_cost(X_features, y_array)
            
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {current_cost:.6f}")
            
            if current_cost < best_cost - min_improvement:
                best_cost = current_cost
                best_weights = np.array(self.weights)
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

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict probability of being AI-generated"""
        if self.weights is None:
            raise ValueError("Model has not been fitted yet!")
        
        # Ensure X has the correct feature names
        if isinstance(X, pd.DataFrame):
            X_features = X[self.feature_names]
        else:
            X_features = pd.DataFrame(X, columns=self.feature_names)
        
        # Scale features and explicitly convert to numpy array
        X_scaled = np.array(self.scaler.transform(X_features))
        
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
        if self.weights is None or not hasattr(self.scaler, 'mean_'):
            raise ValueError("Model has not been fitted yet!")
        
        np.savez(
            path,
            weights=self.weights,
            bias=self.bias,
            scaler_mean=np.array(self.scaler.mean_),
            scaler_scale=np.array(self.scaler.scale_),
            feature_names=self.feature_names
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
    print("\nStarting model training and evaluation...")
    
    # Keep track of the path column
    path_column = df['path']
    
    # Select only the top features
    X = df[LogisticRegression.TOP_FEATURES]
    y = (df['label'] == 'AI').astype(int)
    
    # Split data by class
    X_with_path = pd.concat([X, path_column, pd.Series(y.values, index=X.index, name='label')], axis=1)
    ai_samples = X_with_path[X_with_path['label'] == 1]
    human_samples = X_with_path[X_with_path['label'] == 0]
    
    # Split each class with 80:20 ratio
    ai_train, ai_test = train_test_split(ai_samples, test_size=0.2, random_state=42)
    human_train, human_test = train_test_split(human_samples, test_size=0.2, random_state=42)
    
    # Combine the datasets
    train_data = pd.concat([ai_train, human_train])
    test_data = pd.concat([ai_test, human_test])
    
    # Shuffle the combined datasets
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate features, labels, and paths
    X_train = train_data[LogisticRegression.TOP_FEATURES]
    y_train = train_data['label']
    
    X_test = test_data[LogisticRegression.TOP_FEATURES]
    y_test = test_data['label']
    test_paths = test_data['path']
    
    print(f"\nData split:")
    print(f"Training samples: {len(X_train)} (AI: {sum(y_train == 1)}, Human: {sum(y_train == 0)})")
    print(f"Test samples: {len(X_test)} (AI: {sum(y_test == 1)}, Human: {sum(y_test == 0)})")
    
    # Initialize and train the model
    model = LogisticRegression()
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Save the trained model weights
    model.save_weights('logistic_model_weights.npz')
    
    # Evaluate on both train and test sets
    print("\nEvaluating model...")
    
    # Training set performance
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Set Performance:")
    print(f"Accuracy: {train_acc:.4f}")
    
    # Test set performance
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Print individual predictions with path and features
    print("\nIndividual Test Predictions:")
    for i in range(min(10, len(X_test))):  # Limit to 10 samples for readability
        path = test_paths.iloc[i]
        features = X_test.iloc[i].to_dict()
        actual = "AI" if y_test.iloc[i] == 1 else "Human"
        predicted = "AI" if y_pred[i] == 1 else "Human"
        probability = y_pred_proba[i]
        
        print(f"\nSample {i+1}:")
        print(f"Path: {path}")
        print(f"Actual: {actual}, Predicted: {predicted}, Probability: {probability:.4f}")
        print("Features:")
        for feat, value in features.items():
            print(f"  {feat}: {value:.4f}")
    
    print(f"\nShowing {min(10, len(X_test))} of {len(X_test)} predictions...")
    
    # Calculate detailed metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred_proba)
    }
    
    print(f"\nTest Set Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    
    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nFeature Importance:")
    importance = model.get_feature_importance()
    for feature, score in importance.items():
        print(f"{feature}: {score:.4f}")
    
    # Save metrics
    np.savez('logistic_model_metrics.npz', **metrics)
    
    # In train_and_evaluate_model, add feature normalization check
    print("\nFeature distributions:")
    for feature in LogisticRegression.TOP_FEATURES:
        ai_mean = X_train[X_train.index.isin(ai_train.index)][feature].mean()
        human_mean = X_train[X_train.index.isin(human_train.index)][feature].mean()
        print(f"{feature}: AI={ai_mean:.4f}, Human={human_mean:.4f}, Diff={abs(ai_mean-human_mean):.4f}")
    
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
        prediction = 'AI-generated' if probability >= 0.5 else 'Human-created'
        
        return {
            'probability': float(probability),
            'prediction': prediction
        }
    except FileNotFoundError:
        raise ValueError("No trained model weights found. Please train the model first.")
    

def output_model_results():
    """
    Output model evaluation results and feature importance.
    Can be called from a Jupyter notebook to display results.
    
    Returns:
        tuple: (metrics, feature_importance) containing evaluation metrics and feature importance dict
    """
    try:
        # Load metrics from saved file
        metrics = np.load('logistic_model_metrics.npz')
        
        # Print evaluation results
        print("\nModel Evaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        
        # Load model to get feature importance
        model = LogisticRegression()
        model.load_weights('logistic_model_weights.npz')
        
        # Get and print feature importance
        importance = model.get_feature_importance()
        print("\nFeature Importance:")
        for feature, score in importance.items():
            print(f"{feature}: {score:.4f}")
        
        # Return metrics and importance for further analysis
        return dict(metrics), importance
        
    except FileNotFoundError:
        print("No model metrics or weights found. Please train the model first.")
        return None, None

if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    df = load_or_process_dataset()
    
    # Train and evaluate the model
    print("\nTraining and evaluating model...")
    model, (X_train, X_test, y_train, y_test) = train_and_evaluate_model(df)
    
    # Print evaluation results
    print("\nFinal Evaluation Results:")
    metrics = np.load('logistic_model_metrics.npz')
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    
    # Print feature importance
    print("\nFeature Importance:")
    importance = model.get_feature_importance()
    for feature, score in importance.items():
        print(f"{feature}: {score:.4f}")
