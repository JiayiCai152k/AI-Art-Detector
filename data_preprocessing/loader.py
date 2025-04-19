import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomImageDataset(Dataset):
    """
    Custom Dataset class for lazy loading of images.
    Only loads images when they are accessed, making it memory efficient.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        """
        Args:
            df: DataFrame containing image paths and labels
            transform: Optional transform to be applied on a sample
        """
        self.df = df
        self.transform = (
            transform
            if transform is not None
            else transforms.Compose(
                [
                    transforms.ToTensor(),  # Converts to [0, 1] range and (C, H, W) format
                ]
            )
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path and label
        img_path = self.df.iloc[idx]["path"]
        label = 1 if self.df.iloc[idx]["label"] == "ai" else 0

        # Load and transform image only when accessed
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def load_ukiyo_e_dataset(data_dir="data"):
    # Get paths for both human and AI images
    human_dir = os.path.join(data_dir, "Human_Ukiyo_e")
    ai_dir = os.path.join(data_dir, "AI_SD_ukiyo-e")

    # Get all file paths and dimensions
    def get_files_with_dimensions(directory):
        files_info = []
        for f in os.listdir(directory):
            if f.endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(directory, f)
                with Image.open(path) as img:
                    width, height = img.size
                    files_info.append({"path": path, "width": width, "height": height})
        return files_info

    human_files_info = get_files_with_dimensions(human_dir)
    ai_files_info = get_files_with_dimensions(ai_dir)

    # Create DataFrames with dimensions
    human_df = pd.DataFrame(human_files_info)
    human_df["label"] = "human"

    ai_df = pd.DataFrame(ai_files_info)
    ai_df["label"] = "ai"

    # Combine datasets
    df = pd.concat([human_df, ai_df], ignore_index=True)

    # Print dataset statistics
    print("Dataset Statistics:")
    print(f"Total images: {len(df)}")
    print(f"Human images: {len(human_df)}")
    print(f"AI images: {len(ai_df)}")

    print("\nImage Dimensions Statistics:")
    print("\nHuman Images:")
    print(
        f"Width  - Mean: {human_df['width'].mean():.0f}, Min: {human_df['width'].min()}, Max: {human_df['width'].max()}"
    )
    print(
        f"Height - Mean: {human_df['height'].mean():.0f}, Min: {human_df['height'].min()}, Max: {human_df['height'].max()}"
    )

    print("\nAI Images:")
    print(
        f"Width  - Mean: {ai_df['width'].mean():.0f}, Min: {ai_df['width'].min()}, Max: {ai_df['width'].max()}"
    )
    print(
        f"Height - Mean: {ai_df['height'].mean():.0f}, Min: {ai_df['height'].min()}, Max: {ai_df['height'].max()}"
    )

    # Load and display random samples from each class with dimensions
    sample_human_path = human_df.sample(n=1).iloc[0]["path"]
    sample_ai_path = ai_df.sample(n=1).iloc[0]["path"]

    sample_human = Image.open(sample_human_path)
    sample_ai = Image.open(sample_ai_path)

    # Display samples
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(sample_human)
    plt.title(f"Human Ukiyo-e Sample\nDimensions: {sample_human.size}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sample_ai)
    plt.title(f"AI Generated Ukiyo-e Sample\nDimensions: {sample_ai.size}")
    plt.axis("off")

    plt.show()

    return df
