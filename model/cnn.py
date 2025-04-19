import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import platform


class CNN(nn.Module):
    def save_weights(self):
        path = f"cnn_weights_epoch_{self.num_epochs}_lr_{self.learning_rate}_bs_{self.batch_size}.pth"
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def __init__(
        self,
        learning_rate=0.0001,
        num_epochs=10,
        batch_size=3,
        load_weights_path=None,
    ):
        super(CNN, self).__init__()
        if load_weights_path:
            self.load_weights(load_weights_path)

        # Architecture parameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size  # Smaller batch size due to large images
        self.input_shape = (768, 768, 3)  # Original large input size: 768 x 768 x 3

        # Determine device
        self.device = self._get_device()

        # Convolutional layers with more aggressive downsampling
        self.conv_layers = nn.Sequential(
            # Initial aggressive downsampling
            nn.Conv2d(
                3, 16, kernel_size=7, stride=2, padding=3
            ),  # Output: 384 x 384 x 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  # Output: 96 x 96 x 16
            # Further processing with smaller kernels
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Output: 96 x 96 x 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  # Output: 24 x 24 x 32
            nn.Conv2d(32, 48, kernel_size=3, padding=1),  # Output: 24 x 24 x 48
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 12 x 12 x 48
            nn.Conv2d(48, 64, kernel_size=3, padding=1),  # Output: 12 x 12 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Final output: 6 x 6 x 64
        )

        # Calculate the size of flattened features
        h_out = 6  # Final height
        w_out = 6  # Final width
        flattened_size = 64 * h_out * w_out  # 6 * 6 * 64 = 2,304 features

        # Memory-efficient fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 256),  # From 2,304 -> 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),  # From 256 -> 64
            nn.ReLU(),
            nn.Linear(64, 1),  # Final classification
            nn.Sigmoid(),
        )

        # Initialize weights
        self._initialize_weights()

        # Store activations for attention analysis
        self.activations = {}

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

        # Store input activation

        # Pass through convolutional layers
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = self.fc_layers(x)

        return x

    def fit(self, train_df, val_df=None):
        """
        Train the model using the provided DataFrames containing image paths and labels.
        Compatible with Metal Performance Shaders (MPS) for Mac devices.
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

            # Force garbage collection between epochs
            gc.collect()
            if self.device.type == "mps":
                torch.mps.empty_cache()

        return costs

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

        all_probs = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
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
