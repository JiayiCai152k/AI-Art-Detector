#!/usr/bin/env python3
"""
Fix function to correctly load the dataset for the AI Art Detector project.
"""

import pandas as pd
import os
from pathlib import Path
from data_preprocessing.analyze import analyze_dataset

def load_dataset(force_reprocess=False):
    """
    Correctly load the dataset, either from cache or by processing images.
    
    Args:
        force_reprocess: If True, regenerate the dataset from images
    
    Returns:
        DataFrame: The loaded dataset
    """
    cache_path = 'outputs/processed_features.csv'
    
    # Check if we should use cache
    if not force_reprocess and os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"Loaded {len(df)} samples from cache")
        
        # Verify if the data looks valid
        if len(df) > 0 and 'label' in df.columns:
            print(f"AI images: {len(df[df['label'] == 'AI'])}")
            print(f"Human images: {len(df[df['label'] == 'Human'])}")
            return df
        else:
            print("Cache appears to be invalid. Regenerating...")
    
    # If we get here, we need to regenerate the dataset
    print("Processing images to generate dataset...")
    
    # Make sure to use the correct data directory path
    data_dir = "./data"  # This is the correct path in your project
    
    # Verify that the data directories exist
    ai_path = os.path.join(data_dir, "AI_SD_ukiyo-e")
    human_path = os.path.join(data_dir, "Human_Ukiyo_e")
    
    if not os.path.exists(ai_path) or not os.path.exists(human_path):
        print(f"ERROR: Data directories not found!")
        print(f"AI path exists: {os.path.exists(ai_path)}")
        print(f"Human path exists: {os.path.exists(human_path)}")
        
        # Fall back to the cache if it exists
        if os.path.exists(cache_path):
            print("Falling back to cached data...")
            df = pd.read_csv(cache_path)
            return df
        else:
            print("No cache available either!")
            return pd.DataFrame()
    
    # Process the dataset
    df = analyze_dataset(data_dir=data_dir)
    
    # Save to cache
    if len(df) > 0:
        print("Saving processed dataset to cache...")
        os.makedirs('outputs', exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Saved {len(df)} samples to cache")
    
    return df

# Example usage
if __name__ == "__main__":
    df = load_dataset(force_reprocess=False)
    print(f"\nDataset loaded successfully: {len(df)} samples")
    
    # Verify columns
    required_cols = ['label', 'mean_brightness', 'std_brightness']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")
    else:
        print("All required columns present!") 