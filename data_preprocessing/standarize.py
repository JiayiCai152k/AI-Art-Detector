from typing import Dict, List, Tuple, TypedDict
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


@dataclass
class ImageMetadata:
    """Schema for image metadata in the dataset"""

    path: str
    width: int
    height: int
    label: str


class DatasetStats(TypedDict):
    """Type definition for dataset statistics"""

    total_images: int
    ai_generated: int
    human_created: int
    training_samples: int
    validation_samples: int
    test_samples: int


class ImageSizeStats(TypedDict):
    """Type definition for image size statistics"""

    AI: List[Tuple[int, int]]
    Human: List[Tuple[int, int]]


def get_dataset_statistics(df: pd.DataFrame) -> DatasetStats:
    """
    Calculate basic statistics about the dataset

    Args:
        df: DataFrame containing image metadata with columns [path, width, height, label]

    Returns:
        Dictionary containing dataset statistics
    """
    total_images = len(df)
    ai_generated = len(df[df["label"] == "ai"])
    human_created = len(df[df["label"] == "human"])

    return {
        "total_images": total_images,
        "ai_generated": ai_generated,
        "human_created": human_created,
    }


def analyze_image_sizes(df: pd.DataFrame) -> ImageSizeStats:
    """
    Analyze image dimensions distribution for both AI and human images

    Args:
        df: DataFrame containing image metadata with columns [path, width, height, label]

    Returns:
        Dictionary containing unique dimension pairs for each class
    """
    # Get unique dimensions for each class
    ai_dims = df[df["label"] == "ai"][["width", "height"]].drop_duplicates()
    human_dims = df[df["label"] == "human"][["width", "height"]].drop_duplicates()

    # Convert to list of tuples
    ai_sizes = list(zip(ai_dims["width"], ai_dims["height"]))
    human_sizes = list(zip(human_dims["width"], human_dims["height"]))

    return {"AI": ai_sizes, "Human": human_sizes}


def get_image_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and return image metadata with standardized format

    Args:
        df: DataFrame containing image metadata with columns [path, width, height, label]

    Returns:
        Processed DataFrame with standardized metadata
    """
    # Create a copy to avoid modifying original
    metadata_df = df.copy()

    # Convert paths to absolute paths if they aren't already
    metadata_df["path"] = metadata_df["path"].apply(lambda x: str(Path(x).absolute()))

    # Ensure datatypes are correct
    metadata_df = metadata_df.astype(
        {"path": str, "width": int, "height": int, "label": str}
    )

    # Add additional metadata columns if needed
    metadata_df["aspect_ratio"] = metadata_df["width"] / metadata_df["height"]
    metadata_df["resolution"] = metadata_df["width"] * metadata_df["height"]

    return metadata_df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a comprehensive summary of the dataset

    Args:
        df: DataFrame containing image metadata
    """
    stats = get_dataset_statistics(df)

    print("=== Dataset Summary ===")
    print(f"\nTotal Images: {stats['total_images']}")
    print(f"AI Generated: {stats['ai_generated']}")
    print(f"Human Created: {stats['human_created']}")

    print("\nDataset Statistics:")
    print(df.describe())


if __name__ == "__main__":
    # Example usage
    from data_loading import load_ukiyo_e_dataset  # Import your data loading function

    # Load the dataset
    df = load_ukiyo_e_dataset()

    # Process and analyze
    metadata_df = get_image_metadata(df)
    print_dataset_summary(metadata_df)


def standardize_image_size(
    image_path: str, output_path: str, target_size: Tuple[int, int] = (768, 768)
) -> None:
    """
    Load and resize an image to target size while preserving aspect ratio with padding

    Args:
        image_path: Path to source image
        output_path: Path to save processed image
        target_size: Desired output size (width, height)
    """
    # Load image
    img = Image.open(image_path).convert("RGB")

    # Calculate new dimensions that preserve aspect ratio
    width, height = img.size
    aspect_ratio = width / height

    if aspect_ratio > 1:  # Wider than tall
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:  # Taller than wide
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    # Resize image preserving aspect ratio
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Create a new blank image with target dimensions
    padded_img = Image.new("RGB", target_size, color=(0, 0, 0))

    # Paste the resized image centered on the padded image
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    padded_img.paste(img, (paste_x, paste_y))

    # Save processed image
    padded_img.save(output_path, quality=95)


def standardize_image_crop(
    image_path: str, output_path: str, target_size: Tuple[int, int] = (768, 768)
) -> None:
    """
    Load and resize an image to target size using center cropping approach

    Args:
        image_path: Path to source image
        output_path: Path to save processed image
        target_size: Desired output size (width, height)
    """
    # Load image
    img = Image.open(image_path).convert("RGB")

    # Calculate dimensions to resize to before cropping
    width, height = img.size
    aspect_ratio = width / height

    # Resize so that smallest dimension is at least target size
    if width < height:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    # Resize image
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Calculate crop box for center crop
    left = (new_width - target_size[0]) // 2
    top = (new_height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]

    # Perform center crop
    img = img.crop((left, top, right, bottom))

    # Save processed image
    img.save(output_path, quality=95)


# Given a dataframe, sample one random image from each class and render them side by side
def sample_random_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample one random image from each class in the dataset

    Args:
        df: DataFrame containing image metadata with columns [path, width, height, label]

    Returns:
        DataFrame containing one random image from each class
    """
    # Sample one random image from each class
    ai_df = df[df["label"] == "ai"].sample(n=1)
    human_df = df[df["label"] == "human"].sample(n=1)

    # Render them side by side

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ai_img = Image.open(ai_df["path"].iloc[0])
    human_img = Image.open(human_df["path"].iloc[0])
    axes[0].imshow(ai_img)
    axes[1].imshow(human_img)
    plt.show()
    return pd.concat([ai_df, human_df])


def process_dataset(
    df: pd.DataFrame,
    target_size: Tuple[int, int] = (768, 768),
    technique: str = "aspect_ratio_padding",
) -> pd.DataFrame:
    """
    Process entire dataset to standardized size while preserving aspect ratio

    Args:
        df: DataFrame containing image metadata with columns [path, width, height, label]
        output_dir: Directory to save processed images
        target_size: Desired output size for all images (width, height)

    Returns:
        Updated DataFrame with new image paths and dimensions
    """

    # Find where the script is located, go back one directory, navigate to data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")

    dir_name = (
        f"Human_{technique}" if df["label"].iloc[0] == "human" else f"AI_{technique}"
    )

    # Create new DataFrame for processed data
    processed_df = df.copy()
    processed_df["original_path"] = processed_df["path"]
    processed_df["original_width"] = processed_df["width"]
    processed_df["original_height"] = processed_df["height"]
    processed_df["target_label"] = processed_df["label"].apply(
        lambda x: 1 if x == "ai" else 0
    )

    processed_df["path"] = processed_df["path"].apply(
        lambda x: os.path.join(
            data_dir,
            dir_name,
            os.path.basename(x)
            if df["label"].iloc[0] == "human"
            else os.path.basename(x),
        )
    )
    processed_df["original_aspect_ratio"] = (
        processed_df["width"] / processed_df["height"]
    )
    # IF the folder already exists, skip the processing
    if os.path.exists(os.path.join(data_dir, dir_name)):
        processed_df["width"] = target_size[0]
        processed_df["height"] = target_size[1]
        print(f"Folder {dir_name} already exists, skipping processing")
        return processed_df
    print(
        f"Processing {len(df)} images to {target_size[0]}x{target_size[1]} with preserved aspect ratio..."
    )

    # Create output directories
    os.makedirs(os.path.join(data_dir, dir_name), exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Determine output path
        filename = os.path.basename(row["path"])
        output_path = os.path.join(data_dir, dir_name, filename)
        # Process image
        if technique == "aspect_ratio_padding":
            standardize_image_size(row["path"], output_path, target_size)
        elif technique == "center_crop":
            standardize_image_crop(row["path"], output_path, target_size)

        # Update DataFrame
        processed_df.at[idx, "path"] = output_path
        processed_df.at[idx, "width"] = target_size[0]
        processed_df.at[idx, "height"] = target_size[1]

    print("Processing complete!")

    # Add processing metadata
    processed_df["is_processed"] = True
    processed_df["target_size"] = f"{target_size[0]}x{target_size[1]}"
    processed_df["preservation_method"] = "aspect_ratio_padding"

    return processed_df


if __name__ == "__main__":
    from data_loading import load_ukiyo_e_dataset

    # Load the dataset
    df = load_ukiyo_e_dataset()

    # Process images to standard size while preserving aspect ratio
    processed_df = process_dataset(df, output_dir="post_proceed")

    # Verify processing
    # if verify_processed_images(processed_df):
    #     print(
    #         "All images successfully processed to 768x768 with preserved aspect ratio!"
    #     )
    # else:
    #     print("Some images may not have been processed correctly.")

    # Save processed metadata
    processed_df.to_csv("post_proceed_metadata.csv", index=False)

    # Print summary
    print_dataset_summary(processed_df)


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training, validation and test sets while maintaining equal class distribution.
    Returns DataFrames containing image paths and labels for each split.

    Args:
        df: DataFrame containing image metadata
        train_ratio: Proportion of data to use for training (default: 0.7)
        val_ratio: Proportion of data to use for validation (default: 0.15)
        test_ratio: Proportion of data to use for testing (default: 0.15)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df) containing paths and labels for each split
    """
    # Verify ratios sum to 1
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
        raise ValueError("Split ratios must sum to 1")

    # Get counts for each class
    class_counts = df["label"].value_counts()
    min_class_count = class_counts.min()

    print(f"\nOriginal class distribution:")
    for label, count in class_counts.items():
        print(f"{label}: {count} images")

    # Balance classes by downsampling
    balanced_dfs = []
    for label in df["label"].unique():
        class_df = df[df["label"] == label]
        # Downsample to match the size of the smallest class
        balanced_df = class_df.sample(n=min_class_count, random_state=random_state)
        balanced_dfs.append(balanced_df)

    # Combine balanced classes
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    print(
        f"\nBalanced dataset size: {len(balanced_df)} images ({min_class_count} per class)"
    )

    # Calculate split sizes
    total_samples = len(balanced_df)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    # Shuffle the DataFrame
    shuffled_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    # Split into train, validation, and test sets
    train_df = shuffled_df[:train_size]
    val_df = shuffled_df[train_size : train_size + val_size]
    test_df = shuffled_df[train_size + val_size :]

    print(f"\nDataset split summary:")
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")

    # Print class distribution for each split
    for split_name, split_df in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(f"\nClass distribution for {split_name}:")
        class_dist = split_df["label"].value_counts()
        for label, count in class_dist.items():
            print(f"{label}: {count} images")

    return train_df, val_df, test_df


def center_crop_image(
    img: Image.Image, target_size: Tuple[int, int] = (768, 768)
) -> Image.Image:
    """
    Center crop a single PIL Image to target size

    Args:
        img: PIL Image to crop
        target_size: Desired output size (width, height)

    Returns:
        Center cropped PIL Image
    """
    # Calculate dimensions to resize to before cropping
    width, height = img.size
    aspect_ratio = width / height

    # Resize so that smallest dimension is at least target size
    if width < height:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    # Resize image
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Calculate crop box for center crop
    left = (new_width - target_size[0]) // 2
    top = (new_height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]

    # Perform center crop
    return img.crop((left, top, right, bottom))


#
