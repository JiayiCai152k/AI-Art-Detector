import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import os
from typing import Dict, Tuple, List


def analyze_dataset(notebook_path="../ai-artwork-detection.ipynb") -> pd.DataFrame:
    """
    Analyze the dataset from the Jupyter notebook
    """
    try:
        # Read the notebook data
        print(f"Reading data from notebook: {os.path.abspath(notebook_path)}")
        
        # Load the notebook using pandas
        df = pd.read_json(notebook_path)
        
        # Extract the image data and labels
        # Note: You'll need to adjust these column names based on your notebook's structure
        data = []
        
        # Process each image in the dataset
        for idx, row in df.iterrows():
            try:
                img = Image.open(row['image_path'])
                img_array = np.array(img)
                
                data.append({
                    "path": row['image_path'],
                    "label": "AI" if row['is_ai'] else "Human",  # Adjust based on your label column
                    "width": img.size[0],
                    "height": img.size[1],
                    "aspect_ratio": img.size[0] / img.size[1],
                    "size_kb": os.path.getsize(row['image_path']) / 1024,
                    "channels": img_array.shape[2] if len(img_array.shape) > 2 else 1,
                    "mean_brightness": np.mean(img_array),
                    "std_brightness": np.std(img_array)
                })
            except Exception as e:
                print(f"Error processing image {row['image_path']}: {e}")
                
        result_df = pd.DataFrame(data)
        print(f"Successfully processed {len(result_df)} images")
        return result_df
        
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return pd.DataFrame()


def plot_size_distribution(df: pd.DataFrame) -> None:
    """Plot image size distributions"""
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
    plt.savefig('../outputs/size_distribution.png')
    plt.close()


def plot_brightness_analysis(df: pd.DataFrame) -> None:
    """Plot brightness characteristics"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    sns.boxplot(x='label', y='mean_brightness', data=df)
    plt.title('Mean Brightness Distribution')
    
    plt.subplot(122)
    sns.boxplot(x='label', y='std_brightness', data=df)
    plt.title('Brightness Variation Distribution')
    
    plt.tight_layout()
    plt.savefig('../outputs/brightness_analysis.png')
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
    plt.savefig('../outputs/color_distributions.png')
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


def main():
    """Main function to run the analysis"""
    # Create output directory if it doesn't exist
    os.makedirs('../outputs', exist_ok=True)
    
    # Load and analyze dataset
    print("Loading and analyzing dataset...")
    df = analyze_dataset()
    
    if len(df) == 0:
        print("\nError: No data was loaded. Please check:")
        print("1. Are you running the script from the correct directory?")
        print("2. Is your data in the following structure?")
        print("   data/")
        print("   ├── AI_SD_ukiyo-e/")
        print("   └── Human_Ukiyo-e/")
        print(f"3. Current working directory: {os.getcwd()}")
        return
    
    # Generate and save visualizations
    print("\nGenerating visualizations...")
    plot_size_distribution(df)
    plot_brightness_analysis(df)
    plot_sample_color_distributions(df)
    
    # Generate and save summary report
    print("Generating summary report...")
    summary = generate_summary_report(df)
    with open('../outputs/dataset_analysis.txt', 'w') as f:
        f.write(summary)
    
    # Save dataset metadata
    df.to_csv('../outputs/dataset_metadata.csv', index=False)
    print("Analysis complete! Check the outputs directory for results.")


if __name__ == "__main__":
    main()
