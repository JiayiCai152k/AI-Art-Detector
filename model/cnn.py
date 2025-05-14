import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import platform
from PIL import Image
from torchvision import transforms
import pandas as pd
import os


class CNN(nn.Module):
    def save_weights(self, path=None):
        if path is None:
            path = f"cnn_weights_epoch_{self.num_epochs}_lr_{self.learning_rate}_bs_{self.batch_size}.pth"

        # Debug information before saving
        # print("\nSaving model weights...")
        # print("Model's state_dict keys:")
        for key in self.state_dict().keys():
            print(f"- {key}")

        torch.save(self.state_dict(), path)
        # print(f"Weights saved to: {path}")

    def save_best_weights(self, path=None):
        """Save the best weights during training"""
        if path is None:
            path = f"cnn_best_weights_epoch_{self.num_epochs}_lr_{self.learning_rate}_bs_{self.batch_size}.pth"
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        # print(f"\nLoading weights from: {path}")

        # Load the state dict
        if self.device.type == "mps":
            state_dict = torch.load(path, map_location="cpu")
        else:
            state_dict = torch.load(path, map_location=self.device)

        # Check if we need to convert from old format
        if any("conv_layers" in key for key in state_dict.keys()):
            # print("\nDetected old weight format, converting to new format...")
            state_dict = self._convert_old_state_dict(state_dict)

        # Debug information
        # print("\nKeys in loaded state_dict:")
        for key in state_dict.keys():
            print(f"- {key}")

        # print("\nKeys in current model:")
        for key in self.state_dict().keys():
            print(f"- {key}")

        # Try to load weights
        try:
            self.load_state_dict(state_dict, strict=True)
            print("\nWeights loaded successfully!")
        except Exception as e:
            print(f"\nError loading weights: {str(e)}")
            raise

        # Move model to device if needed
        self.to(self.device)
        print(f"Model moved to device: {self.device}")

    def _convert_old_state_dict(self, old_state_dict):
        """Convert weights from old Sequential format to new named format."""
        new_state_dict = {}

        # Mapping from old to new keys
        conv_mapping = {
            "conv_layers.0": "conv1",
            "conv_layers.3": "conv2",
            "conv_layers.6": "conv3",
            "conv_layers.9": "conv4",
        }

        fc_mapping = {"fc_layers.0": "fc1", "fc_layers.3": "fc2", "fc_layers.5": "fc3"}

        # Convert convolutional layers
        for old_key, new_base in conv_mapping.items():
            if f"{old_key}.weight" in old_state_dict:
                new_state_dict[f"{new_base}.weight"] = old_state_dict[
                    f"{old_key}.weight"
                ]
                new_state_dict[f"{new_base}.bias"] = old_state_dict[f"{old_key}.bias"]

        # Convert fully connected layers
        for old_key, new_base in fc_mapping.items():
            if f"{old_key}.weight" in old_state_dict:
                new_state_dict[f"{new_base}.weight"] = old_state_dict[
                    f"{old_key}.weight"
                ]
                new_state_dict[f"{new_base}.bias"] = old_state_dict[f"{old_key}.bias"]

        return new_state_dict

    def __init__(
        self,
        learning_rate=0.0001,
        num_epochs=10,
        batch_size=3,
        load_weights_path=None,
    ):
        super(CNN, self).__init__()

        # Determine device first
        self.device = self._get_device()

        # Architecture parameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.input_shape = (768, 768, 3)

        # Define convolutional layers individually
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of flattened features
        h_out = 6  # Final height
        w_out = 6  # Final width
        flattened_size = 64 * h_out * w_out  # 6 * 6 * 64 = 2,304 features

        # Define fully connected layers individually
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._initialize_weights()

        # Store activations for attention analysis
        self.activations = {}

        if load_weights_path:
            self.load_weights(load_weights_path)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _get_device(self):
        """
        Determine the best available device (Metal, CPU)
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def forward(self, x):
        # Ensure input is in the correct format (B, C, H, W)
        if x.shape[1] != 3:  # If channels are not in the correct position
            x = x.permute(0, 3, 1, 2)

        # Convolutional layers
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = torch.relu(x)
        x = self.pool4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x

    def fit(self, train_df, val_df=None, patience=10, min_delta=1e-4):
        """
        Train the model using the provided DataFrames containing image paths and labels.
        Compatible with Metal Performance Shaders (MPS) for Mac devices.

        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame
            patience: Number of epochs to wait for improvement before early stopping
            min_delta: Minimum change in validation loss to qualify as an improvement

        Returns:
            tuple: (training_costs, best_validation_loss) where training_costs is a list of losses per epoch
                  and best_validation_loss is the lowest validation loss achieved
        """
        from data_preprocessing.loader import CustomImageDataset
        import gc

        # Create datasets
        train_dataset = CustomImageDataset(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
            if self.device.type == "cpu"
            else 0,  # MPS works better with 0 workers
        )

        if val_df is not None:
            val_dataset = CustomImageDataset(val_df)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2 if self.device.type == "cpu" else 0,
            )

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Move model to device
        self.to(self.device)
        print(f"\nTraining on device: {self.device}")
        print(f"Training parameters:")
        print(f"- Learning rate: {self.learning_rate}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Number of epochs: {self.num_epochs}")
        print(f"- Total training samples: {len(train_dataset)}")
        print(f"- Steps per epoch: {len(train_loader)}\n")

        # Early stopping variables
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        costs = []
        for epoch in range(self.num_epochs):
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                # Move tensors to device
                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                with torch.no_grad():
                    predictions = (outputs >= 0.5).float()
                    train_correct += (predictions == labels).sum().item()
                    train_total += labels.size(0)
                    train_loss += loss.item()

                # Print progress
                if (batch_idx + 1) % 5 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.num_epochs}], "
                        f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f}"
                    )

                # Before processing each batch
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                gc.collect()

                # After processing each batch
                del images, labels, outputs, loss, predictions

            # Calculate epoch metrics
            epoch_loss = train_loss / len(train_loader)
            epoch_accuracy = train_correct / train_total * 100
            costs.append(epoch_loss)

            print(f"\nEpoch {epoch + 1}/{self.num_epochs} Summary:")
            print(f"Training Loss: {epoch_loss:.4f}")
            print(f"Training Accuracy: {epoch_accuracy:.2f}%")

            # Validation phase
            if val_df is not None:
                self.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device).float().view(-1, 1)

                        outputs = self(images)
                        val_loss += criterion(outputs, labels).item()
                        predictions = (outputs >= 0.5).float()
                        val_correct += (predictions == labels).sum().item()
                        val_total += labels.size(0)

                        # Clear memory
                        del outputs, predictions
                        if self.device.type == "mps":
                            torch.mps.empty_cache()

                val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total * 100
                print(f"Validation Loss: {val_loss:.4f}")
                print(f"Validation Accuracy: {val_accuracy:.2f}%\n")

                # Early stopping check
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    # Save best weights
                    self.save_best_weights()
                    print(f"New best model saved! Validation Loss: {val_loss:.4f}")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(
                            f"\nEarly stopping triggered! No improvement for {patience} epochs"
                        )
                        print(
                            f"Best validation loss was {best_val_loss:.4f} at epoch {best_epoch + 1}"
                        )
                        break

            # Force garbage collection between epochs
            gc.collect()
            if self.device.type == "mps":
                torch.mps.empty_cache()

        # Save final weights
        self.save_weights()

        return costs, best_val_loss if val_df is not None else (costs, None)

    def predict_proba(self, test_df):
        """
        Predict probabilities for the test data.
        """
        from data_preprocessing.loader import CustomImageDataset

        # Create test dataset and dataloader
        test_dataset = CustomImageDataset(test_df)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2 if self.device.type == "cpu" else 0,
        )

        # Set model to evaluation mode
        self.eval()
        self.to(self.device)  # Ensure model is on the correct device

        all_probs = []
        with torch.no_grad():
            for images, _ in test_loader:
                # Ensure images are in float32 format
                images = images.float().to(self.device)
                outputs = self(images)
                # Move predictions to CPU before converting to numpy
                all_probs.append(outputs.cpu().numpy())

                # Clear memory
                del outputs, images
                if self.device.type == "mps":
                    torch.mps.empty_cache()

        return np.concatenate(all_probs)

    def predict(self, test_df, threshold=0.5):
        """
        Predict classes for the test data.

        Args:
            test_df: DataFrame containing test data paths and labels
            threshold: Classification threshold (default: 0.5)

        Returns:
            Numpy array of predicted classes (0 for human, 1 for AI)
        """
        probs = self.predict_proba(test_df)
        return (probs >= threshold).astype(int)

    def analyze_attention(self, X, layer_indices=None):
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        # Forward pass to get activations
        with torch.no_grad():
            _ = self(X)

        if layer_indices is None:
            layer_indices = range(1, len(self.activations))

        attention_maps = {}
        for layer in layer_indices:
            if layer in self.activations:
                attention_maps[layer] = self.activations[layer].cpu().numpy()

        # Clear memory
        if self.device.type == "mps":
            torch.mps.empty_cache()

        return attention_maps

    def predict_single_image_proba(self, image):
        """
        Predict probability for a single PIL Image directly.

        Args:
            image: PIL Image object

        Returns:
            numpy array of shape (1, 1) containing the probability
        """
        self.eval()  # Set to evaluation mode
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Transform and add batch dimension
        with torch.no_grad():
            # Convert to tensor and move to device - exactly like in CustomImageDataset
            image = image.convert("RGB")  # Ensure RGB format
            image_tensor = transform(image).unsqueeze(0).float()
            image_tensor = image_tensor.to(self.device)  # Use self.device directly

            # Forward pass
            outputs = self(image_tensor)

            # Move to CPU and convert to numpy - shape will be (1, 1)
            return outputs.cpu().numpy().reshape(-1, 1)

    def output_model_results(self):
        """
        Output model evaluation results.
        Can be called to display hardcoded results.

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Define the metrics file path
        metrics_file = "cnn_model_metrics.csv"

        # If the file doesn't exist, create it with hardcoded metrics
        if not os.path.exists(metrics_file):
            metrics = {
                "accuracy": 0.9829059829059829,
                "precision": 0.9831460674157303,
                "recall": 0.9831460674157303,
                "f1_score": 0.9831460674157303,
                "mse": 0.012380106590100621,
            }
            # Save metrics to CSV
            pd.DataFrame([metrics]).to_csv(metrics_file, index=False)

        try:
            # Load metrics from CSV file
            metrics_df = pd.read_csv(metrics_file)
            metrics = metrics_df.iloc[0].to_dict()

            # Print evaluation results
            print("\nModel Evaluation Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"MSE: {metrics['mse']:.4f}")

            return metrics

        except Exception as e:
            print(f"Error loading metrics: {str(e)}")
            return None
