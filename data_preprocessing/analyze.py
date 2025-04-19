import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import os
from typing import Dict, Tuple, List


def analyze_dataset(data_dir="../data") -> pd.DataFrame:
    """
    Analyze the ukiyo-e dataset and return basic statistics
    """
    ai_path = os.path.join(data_dir, "AI_SD_ukiyo-e")
    human_path = os.path.join(data_dir, "Human_Ukiyo_e")
    data = []
    
    # Print absolute paths for debugging
    print(f"Looking for AI images in: {os.path.abspath(ai_path)}")
    print(f"Looking for Human images in: {os.path.abspath(human_path)}")
    
    # Check if directories exist
    if not os.path.exists(ai_path):
        print(f"Error: AI images directory not found at {os.path.abspath(ai_path)}")
        return pd.DataFrame()
    
    if not os.path.exists(human_path):
        print(f"Error: Human images directory not found at {os.path.abspath(human_path)}")
        return pd.DataFrame()
    
    # Count images found
    ai_images = list(Path(ai_path).glob("*.[jp][pn][g]"))
    human_images = list(Path(human_path).glob("*.[jp][pn][g]"))
    
    print(f"Found {len(ai_images)} AI-generated images")
    print(f"Found {len(human_images)} human-created images")
    
    if len(ai_images) == 0 and len(human_images) == 0:
        print("No images found in either directory!")
        return pd.DataFrame()

    # Process AI-generated images
    for img_path in Path(ai_path).glob("*.[jp][pn][g]"):
        try:
            with Image.open(img_path) as img:
                img_array = np.array(img)
                data.append({
                    "path": str(img_path),
                    "label": "AI",
                    "width": img.size[0],
                    "height": img.size[1],
                    "aspect_ratio": img.size[0] / img.size[1],
                    "size_kb": os.path.getsize(img_path) / 1024,
                    "channels": img_array.shape[2] if len(img_array.shape) > 2 else 1,
                    "mean_brightness": np.mean(img_array),
                    "std_brightness": np.std(img_array)
                })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Process human-created images
    for img_path in Path(human_path).glob("*.[jp][pn][g]"):
        try:
            with Image.open(img_path) as img:
                img_array = np.array(img)
                data.append({
                    "path": str(img_path),
                    "label": "Human",
                    "width": img.size[0],
                    "height": img.size[1],
                    "aspect_ratio": img.size[0] / img.size[1],
                    "size_kb": os.path.getsize(img_path) / 1024,
                    "channels": img_array.shape[2] if len(img_array.shape) > 2 else 1,
                    "mean_brightness": np.mean(img_array),
                    "std_brightness": np.std(img_array)
                })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return pd.DataFrame(data)


def plot_size_distribution(df: pd.DataFrame) -> None:
    """Plot image size distributions"""
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Debug information
    print(f"DataFrame shape: {df.shape}")
    print("DataFrame columns:", df.columns.tolist())
    print("\nFirst few rows of data:")
    print(df.head())
    
    if len(df) == 0:
        print("Error: DataFrame is empty, cannot create plots")
        return
    
    if 'label' not in df.columns:
        print("Error: 'label' column not found in DataFrame")
        return
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    sns.boxplot(x='label', y='width', data=df)
    plt.title('Width Distribution')
    
    plt.subplot(132)
    sns.boxplot(x='label', y='height', data=df)
    plt.title('Height Distribution')
    
    plt.subplot(133)
    sns.boxplot(x='label', y='aspect_ratio', data=df)
    plt.title('Aspect Ratio Distribution')
    
    plt.tight_layout()
    plt.savefig('outputs/size_distribution.png')
    plt.close()


def plot_brightness_analysis(df: pd.DataFrame) -> None:
    """Plot brightness characteristics"""
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    sns.boxplot(x='label', y='mean_brightness', data=df)
    plt.title('Mean Brightness Distribution')
    
    plt.subplot(122)
    sns.boxplot(x='label', y='std_brightness', data=df)
    plt.title('Brightness Variation Distribution')
    
    plt.tight_layout()
    plt.savefig('outputs/brightness_analysis.png')
    plt.close()


def analyze_color_distribution(image_path: str) -> Dict[str, np.ndarray]:
    """Analyze color distribution of an image"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    color_features = {}
    for i, color in enumerate(['red', 'green', 'blue']):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        color_features[color] = hist.flatten()
    
    return color_features


def plot_sample_color_distributions(df: pd.DataFrame, samples_per_class: int = 5) -> None:
    """Plot color distributions for sample images from each class"""
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    for idx, label in enumerate(['AI', 'Human']):
        sample_paths = df[df['label'] == label]['path'].sample(samples_per_class)
        
        for i, path in enumerate(sample_paths):
            color_dist = analyze_color_distribution(path)
            plt.subplot(2, samples_per_class, i + 1 + idx * samples_per_class)
            
            for color, hist in color_dist.items():
                plt.plot(hist, color=color, alpha=0.7)
            
            plt.title(f'{label} Sample {i+1}')
            plt.xticks([])
    
    plt.tight_layout()
    plt.savefig('outputs/color_distributions.png')
    plt.close()


def generate_summary_report(df: pd.DataFrame) -> str:
    """Generate a detailed summary report of the dataset"""
    summary = f"""
    Ukiyo-e Dataset Analysis Summary
    ==============================

    Dataset Composition:
    ------------------
    Total Images: {len(df)}
    AI-generated: {len(df[df['label'] == 'AI'])}
    Human-created: {len(df[df['label'] == 'Human'])}

    Image Dimensions:
    ---------------
    Width (pixels):
        AI     - Mean: {df[df['label'] == 'AI']['width'].mean():.1f}, Std: {df[df['label'] == 'AI']['width'].std():.1f}
        Human  - Mean: {df[df['label'] == 'Human']['width'].mean():.1f}, Std: {df[df['label'] == 'Human']['width'].std():.1f}
    
    Height (pixels):
        AI     - Mean: {df[df['label'] == 'AI']['height'].mean():.1f}, Std: {df[df['label'] == 'AI']['height'].std():.1f}
        Human  - Mean: {df[df['label'] == 'Human']['height'].mean():.1f}, Std: {df[df['label'] == 'Human']['height'].std():.1f}
    
    Aspect Ratio:
        AI     - Mean: {df[df['label'] == 'AI']['aspect_ratio'].mean():.2f}, Std: {df[df['label'] == 'AI']['aspect_ratio'].std():.2f}
        Human  - Mean: {df[df['label'] == 'Human']['aspect_ratio'].mean():.2f}, Std: {df[df['label'] == 'Human']['aspect_ratio'].std():.2f}

    File Sizes:
    ----------
    AI     - Mean: {df[df['label'] == 'AI']['size_kb'].mean():.1f}KB, Std: {df[df['label'] == 'AI']['size_kb'].std():.1f}KB
    Human  - Mean: {df[df['label'] == 'Human']['size_kb'].mean():.1f}KB, Std: {df[df['label'] == 'Human']['size_kb'].std():.1f}KB

    Brightness Characteristics:
    ------------------------
    Mean Brightness:
        AI     - Mean: {df[df['label'] == 'AI']['mean_brightness'].mean():.1f}, Std: {df[df['label'] == 'AI']['mean_brightness'].std():.1f}
        Human  - Mean: {df[df['label'] == 'Human']['mean_brightness'].mean():.1f}, Std: {df[df['label'] == 'Human']['mean_brightness'].std():.1f}
    """
    return summary

# Add this at the bottom of the file
__all__ = [
    'analyze_dataset',
    'plot_size_distribution',
    'plot_brightness_analysis',
    'plot_sample_color_distributions',
    'generate_summary_report'
]