from typing import Dict, List, Tuple, TypedDict
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


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

    # Calculate suggested split sizes (70/15/15)
    training_samples = int(total_images * 0.7)
    validation_samples = int(total_images * 0.15)
    test_samples = total_images - training_samples - validation_samples

    return {
        "total_images": total_images,
        "ai_generated": ai_generated,
        "human_created": human_created,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
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
    size_stats = analyze_image_sizes(df)

    print("=== Dataset Summary ===")
    print(f"\nTotal Images: {stats['total_images']}")
    print(f"AI Generated: {stats['ai_generated']}")
    print(f"Human Created: {stats['human_created']}")

    print("\nSuggested Split:")
    print(f"Training: {stats['training_samples']}")
    print(f"Validation: {stats['validation_samples']}")
    print(f"Test: {stats['test_samples']}")

    print("\nUnique Image Dimensions:")
    print("AI Generated Images:")
    for dims in size_stats["AI"]:
        print(f"  {dims[0]}x{dims[1]}")
    print("Human Created Images:")
    for dims in size_stats["Human"]:
        print(f"  {dims[0]}x{dims[1]}")

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
