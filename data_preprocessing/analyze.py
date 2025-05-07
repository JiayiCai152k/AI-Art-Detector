import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import os
from typing import Dict, Tuple, List, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats


def extract_color_features(img_array: np.ndarray) -> Dict[str, float]:
    """Extract comprehensive color-based features from an image array"""
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Convert to different color spaces
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # RGB features
    rgb_means = np.mean(img_array, axis=(0, 1))
    rgb_stds = np.std(img_array, axis=(0, 1))
    
    # HSV features
    hsv_means = np.mean(hsv_img, axis=(0, 1))
    hsv_stds = np.std(hsv_img, axis=(0, 1))
    
    # Calculate saturation metrics
    saturation = hsv_img[:, :, 1]
    
    # Calculate hue distribution
    hue_hist = np.histogram(hsv_img[:, :, 0], bins=8, range=(0, 180))[0]
    hue_dist = hue_hist / np.sum(hue_hist)
    
    # Brightness and contrast from LAB space
    brightness = np.mean(lab_img[:, :, 0])
    contrast = np.std(lab_img[:, :, 0])
    
    features = {
        # RGB features
        'red_mean': rgb_means[0],
        'green_mean': rgb_means[1],
        'blue_mean': rgb_means[2],
        'red_std': rgb_stds[0],
        'green_std': rgb_stds[1],
        'blue_std': rgb_stds[2],
        'red_ratio': rgb_means[0] / np.sum(rgb_means),
        'green_ratio': rgb_means[1] / np.sum(rgb_means),
        'blue_ratio': rgb_means[2] / np.sum(rgb_means),
        
        # HSV features
        'hue_mean': hsv_means[0],
        'saturation_mean': hsv_means[1],
        'value_mean': hsv_means[2],
        'hue_std': hsv_stds[0],
        'saturation_std': hsv_stds[1],
        'value_std': hsv_stds[2],
        
        # Saturation metrics
        'saturation_median': np.median(saturation),
        'saturation_quartile_1': np.percentile(saturation, 25),
        'saturation_quartile_3': np.percentile(saturation, 75),
        
        # Hue distribution features
        **{f'hue_bin_{i}': val for i, val in enumerate(hue_dist)},
        
        # LAB color space features
        'lab_brightness': brightness,
        'lab_contrast': contrast,
    }
    
    return features


def extract_texture_features(img_array: np.ndarray) -> Dict[str, float]:
    """Extract texture-based features from an image array"""
    # Convert to grayscale if needed
    if len(img_array.shape) > 2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Calculate texture features
    features = {
        'entropy': -np.sum(np.multiply(gray/255, np.log2(gray/255 + 1e-10))),
        'contrast': np.std(gray),
    }
    
    # Edge features
    edges = cv2.Canny(gray, 100, 200)
    features['edge_density'] = np.mean(edges > 0)
    
    return features


def analyze_dataset(data_dir="./data") -> pd.DataFrame:
    """
    Analyze the ukiyo-e dataset and return basic statistics with engineered features
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

    def process_single_image(img_path: Path, label: str) -> Dict[str, Any]:
        """Process a single image and extract all features"""
        try:
            with Image.open(img_path) as img:
                img_array = np.array(img)
                
                # Basic features
                features = {
                    "path": str(img_path),
                    "label": label,
                    #"width": img.size[0],
                    #"height": img.size[1],
                    "aspect_ratio": img.size[0] / img.size[1],
                    #"size_kb": os.path.getsize(img_path) / 1024,
                    "channels": img_array.shape[2] if len(img_array.shape) > 2 else 1,
                    "mean_brightness": float(np.mean(img_array)),
                    "std_brightness": float(np.std(img_array))
                }
                
                # Convert to uint8 for feature extraction if needed
                img_array_uint8 = img_array
                if img_array.dtype != np.uint8:
                    img_array_uint8 = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
                
                # Add color features
                color_features = extract_color_features(img_array_uint8)
                features.update(color_features)
                
                # Add texture features
                texture_features = extract_texture_features(img_array_uint8)
                features.update(texture_features)
                
                # Add line features
                line_features = extract_line_features(img_array_uint8)
                features.update(line_features)
                
                # Add contrast features
                contrast_features = extract_contrast_features(img_array_uint8)
                features.update(contrast_features)
                
                print(f"✓ Processed {label} image: {os.path.basename(img_path)}")
                return features
                
        except Exception as e:
            print(f"✗ Error processing {img_path}: {e}")
            return {}

    # Process AI-generated images
    for img_path in Path(ai_path).glob("*.[jp][pn][g]"):
        features = process_single_image(img_path, "AI")
        if features:
            data.append(features)

    # Process human-created images
    for img_path in Path(human_path).glob("*.[jp][pn][g]"):
        features = process_single_image(img_path, "Human")
        if features:
            data.append(features)
    
    df = pd.DataFrame(data)
    
    # Add feature descriptions
    feature_descriptions = {
        'width': 'Image width in pixels',
        'height': 'Image height in pixels',
        'aspect_ratio': 'Width/Height ratio',
        'size_kb': 'File size in kilobytes',
        'channels': 'Number of color channels',
        'mean_brightness': 'Average pixel intensity',
        'std_brightness': 'Standard deviation of pixel intensity',
        'red_mean': 'Mean value of red channel',
        'green_mean': 'Mean value of green channel',
        'blue_mean': 'Mean value of blue channel',
        'red_std': 'Standard deviation of red channel',
        'green_std': 'Standard deviation of green channel',
        'blue_std': 'Standard deviation of blue channel',
        'red_ratio': 'Proportion of red in total RGB',
        'green_ratio': 'Proportion of green in total RGB',
        'blue_ratio': 'Proportion of blue in total RGB',
        'entropy': 'Image entropy (measure of randomness)',
        'contrast': 'Standard deviation of grayscale values',
        'edge_density': 'Proportion of edge pixels'
    }
    
    df.attrs['feature_descriptions'] = feature_descriptions
    return df


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

    Color Characteristics:
    -------------------
    RGB Means:
        AI     - R: {df[df['label'] == 'AI']['red_mean'].mean():.1f}, G: {df[df['label'] == 'AI']['green_mean'].mean():.1f}, B: {df[df['label'] == 'AI']['blue_mean'].mean():.1f}
        Human  - R: {df[df['label'] == 'Human']['red_mean'].mean():.1f}, G: {df[df['label'] == 'Human']['green_mean'].mean():.1f}, B: {df[df['label'] == 'Human']['blue_mean'].mean():.1f}
    
    Color Ratios:
        AI     - R: {df[df['label'] == 'AI']['red_ratio'].mean():.3f}, G: {df[df['label'] == 'AI']['green_ratio'].mean():.3f}, B: {df[df['label'] == 'AI']['blue_ratio'].mean():.3f}
        Human  - R: {df[df['label'] == 'Human']['red_ratio'].mean():.3f}, G: {df[df['label'] == 'Human']['green_ratio'].mean():.3f}, B: {df[df['label'] == 'Human']['blue_ratio'].mean():.3f}

    Texture and Line Features:
    -----------------------
    Edge Density:
        AI     - Mean: {df[df['label'] == 'AI']['edge_density'].mean():.3f}, Std: {df[df['label'] == 'AI']['edge_density'].std():.3f}
        Human  - Mean: {df[df['label'] == 'Human']['edge_density'].mean():.3f}, Std: {df[df['label'] == 'Human']['edge_density'].std():.3f}
    
    Line Statistics:
        AI     - Count: {df[df['label'] == 'AI']['line_count'].mean():.1f}, Density: {df[df['label'] == 'AI']['line_density'].mean():.4f}
        Human  - Count: {df[df['label'] == 'Human']['line_count'].mean():.1f}, Density: {df[df['label'] == 'Human']['line_density'].mean():.4f}

    Contrast Analysis:
    ---------------
    Michelson Contrast:
        AI     - Mean: {df[df['label'] == 'AI']['michelson_contrast'].mean():.3f}, Std: {df[df['label'] == 'AI']['michelson_contrast'].std():.3f}
        Human  - Mean: {df[df['label'] == 'Human']['michelson_contrast'].mean():.3f}, Std: {df[df['label'] == 'Human']['michelson_contrast'].std():.3f}
    
    RMS Contrast:
        AI     - Mean: {df[df['label'] == 'AI']['rms_contrast'].mean():.3f}, Std: {df[df['label'] == 'AI']['rms_contrast'].std():.3f}
        Human  - Mean: {df[df['label'] == 'Human']['rms_contrast'].mean():.3f}, Std: {df[df['label'] == 'Human']['rms_contrast'].std():.3f}

    Entropy and Complexity:
    --------------------
    Image Entropy:
        AI     - Mean: {df[df['label'] == 'AI']['entropy'].mean():.3f}, Std: {df[df['label'] == 'AI']['entropy'].std():.3f}
        Human  - Mean: {df[df['label'] == 'Human']['entropy'].mean():.3f}, Std: {df[df['label'] == 'Human']['entropy'].std():.3f}
    """
    return summary


def load_or_process_dataset(force_reprocess=False, data_dir="./data") -> pd.DataFrame:
    """Load dataset from cache or process it if needed"""
    cache_path = 'outputs/processed_features.csv'
    os.makedirs('outputs', exist_ok=True)
    
    if not force_reprocess and os.path.exists(cache_path):
        print("Loading cached dataset...")
        df = pd.read_csv(cache_path)
        print(f"Loaded {len(df)} samples from cache")
        return df
    
    print("Processing dataset from images...")
    df = analyze_dataset(data_dir)
    
    # Save to cache
    print("Saving processed dataset to cache...")
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} samples to {cache_path}")
    
    return df


def analyze_feature_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze statistical differences between AI and human images for each feature"""
    feature_stats = []
    
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    features_to_analyze = [f for f in numeric_features if f not in ['path', 'label']]
    
    for feature in features_to_analyze:
        ai_values = np.array(df[df['label'] == 'AI'][feature].tolist(), dtype=np.float64)
        human_values = np.array(df[df['label'] == 'Human'][feature].tolist(), dtype=np.float64)
        
        # Perform t-test with nan_policy='omit' to handle NaN values
        t_stat, p_value = stats.ttest_ind(ai_values, human_values, nan_policy='omit')
        p_val = float(np.asarray(p_value, dtype=np.float64).item())
        
        # Calculate pooled standard deviation with safety checks
        n1, n2 = ai_values.size, human_values.size
        var1, var2 = np.nanvar(ai_values), np.nanvar(human_values)
        
        # Handle potential division by zero in Cohen's d calculation
        try:
            pooled_std = np.sqrt(
                ((n1 - 1) * var1 + (n2 - 1) * var2) / float(n1 + n2 - 2)
            )
            
            # Only calculate Cohen's d if pooled_std is not too close to zero
            if pooled_std > 1e-10:
                cohens_d = float(np.nanmean(ai_values) - np.nanmean(human_values)) / pooled_std
            else:
                cohens_d = 0.0  # If standard deviation is too small, effect size is meaningless
        except (ZeroDivisionError, RuntimeWarning):
            cohens_d = 0.0
            
        feature_stats.append({
            'feature': feature,
            'ai_mean': float(np.nanmean(ai_values)),
            'human_mean': float(np.nanmean(human_values)),
            'difference': float(abs(np.nanmean(ai_values) - np.nanmean(human_values))),
            'p_value': float(p_val) if not np.isnan(p_val) else 1.0,
            'effect_size': float(abs(cohens_d))
        })
    
    return pd.DataFrame(feature_stats).sort_values('effect_size', ascending=False)


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Plot correlation matrix of features"""
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    features_to_analyze = [f for f in numeric_features if f not in ['path', 'label']]
    
    corr_matrix = df[features_to_analyze].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('outputs/correlation_matrix.png')
    plt.close()


def perform_pca_analysis(df: pd.DataFrame) -> Tuple[PCA, np.ndarray]:
    """Perform PCA analysis on the features"""
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    features_for_pca = [f for f in numeric_features if f not in ['path', 'label']]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_for_pca])
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True)
    plt.savefig('outputs/pca_explained_variance.png')
    plt.close()
    
    # Plot first two principal components
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=pd.factorize(df['label'])[0], 
                         cmap='viridis', alpha=0.6)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA: First Two Principal Components')
    plt.colorbar(scatter, label='Class')
    plt.savefig('outputs/pca_visualization.png')
    plt.close()
    
    return pca, X_pca


def plot_feature_distributions(df: pd.DataFrame, feature_analysis: pd.DataFrame) -> None:
    """Plot distributions of top features"""
    top_features = feature_analysis['feature'].head(6).tolist()
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(data=df, x=feature, hue='label')
        plt.title(f'{feature} Distribution')
    
    plt.tight_layout()
    plt.savefig('outputs/feature_distributions.png')
    plt.close()


def get_top_correlations(df: pd.DataFrame, n: int = 5) -> List[Tuple[Tuple[str, str], float]]:
    """Get top n correlated feature pairs"""
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    features_to_analyze = [f for f in numeric_features if f not in ['path', 'label']]
    corr_matrix = df[features_to_analyze].corr()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    series = upper.unstack()
    values = np.array(series.values, dtype=float)
    idx = np.argsort(values)[::-1]
    series = pd.Series(values[idx], index=series.index[idx])
    series = series.dropna()
    
    return list(zip(zip(*[series.index.get_level_values(i) for i in [0, 1]]), 
                    series.values))[:n]


def generate_analysis_report(df: pd.DataFrame, feature_analysis: pd.DataFrame, pca: PCA) -> str:
    """Generate a comprehensive analysis report"""
    report = """
    Feature Analysis Report
    ======================
    
    Dataset Overview:
    ----------------
    Total Images: {}
    AI-generated: {}
    Human-created: {}
    Total Features: {}
    
    Top Distinguishing Features:
    --------------------------
    {}
    
    PCA Analysis:
    ------------
    Components needed for 90% variance: {}
    First component explained variance: {:.2f}%
    Second component explained variance: {:.2f}%
    
    Key Findings:
    ------------
    1. Most significant features (by effect size):
       {}
    
    2. Highly correlated feature pairs:
       {}
    """.format(
        len(df),
        len(df[df['label'] == 'AI']),
        len(df[df['label'] == 'Human']),
        len(feature_analysis),
        feature_analysis[['feature', 'effect_size', 'p_value']].head().to_string(),
        np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1,
        pca.explained_variance_ratio_[0] * 100,
        pca.explained_variance_ratio_[1] * 100,
        ", ".join(feature_analysis['feature'].head().tolist()),
        "\n       ".join([f"{pair[0]} - {pair[1]}: {corr:.2f}" 
                         for pair, corr in get_top_correlations(df, 5)])
    )
    
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/analysis_report.txt', 'w') as f:
        f.write(report)
    
    return report


def analyze_features(df: pd.DataFrame) -> None:
    """Run complete feature analysis pipeline"""
    print("Analyzing feature differences...")
    feature_analysis = analyze_feature_differences(df)
    print("\nTop 10 most distinguishing features:")
    print(feature_analysis.head(10))

    print("\nGenerating correlation matrix...")
    plot_correlation_matrix(df)

    print("\nPerforming PCA analysis...")
    pca, X_pca = perform_pca_analysis(df)

    print("\nPlotting feature distributions...")
    plot_feature_distributions(df, feature_analysis)

    print("\nGenerating analysis report...")
    report = generate_analysis_report(df, feature_analysis, pca)
    print("\nAnalysis complete! Check the 'outputs' directory for detailed visualizations and reports.")
    print("\nKey findings summary:")
    print(report)


def extract_line_features(img_array: np.ndarray) -> Dict[str, float]:
    """Extract line-based features from an image"""
    # Convert to grayscale if needed
    if len(img_array.shape) > 2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Line segment detection using LSD
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=30, maxLineGap=10)
    
    # Get image dimensions safely
    height, width = gray.shape[:2]
    image_area = float(height * width)
    
    if lines is not None:
        # Calculate line statistics
        line_lengths = []
        line_angles = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            line_lengths.append(length)
            line_angles.append(angle)
        
        features = {
            'line_count': float(len(lines)),
            'avg_line_length': float(np.mean(line_lengths)),
            'std_line_length': float(np.std(line_lengths)),
            'median_line_length': float(np.median(line_lengths)),
            'line_density': float(len(lines)) / image_area,
            'horizontal_lines': float(np.sum(np.abs(np.array(line_angles)) < 10)),
            'vertical_lines': float(np.sum(np.abs(np.array(line_angles) - 90) < 10)),
        }
    else:
        features = {
            'line_count': 0.0,
            'avg_line_length': 0.0,
            'std_line_length': 0.0,
            'median_line_length': 0.0,
            'line_density': 0.0,
            'horizontal_lines': 0.0,
            'vertical_lines': 0.0,
        }
    
    return features


def extract_contrast_features(img_array: np.ndarray) -> Dict[str, float]:
    """Extract detailed contrast features from an image"""
    if len(img_array.shape) > 2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Local contrast using different window sizes
    local_contrast_3x3 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    local_contrast_5x5 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    
    # Michelson contrast
    min_val = np.min(gray)
    max_val = np.max(gray)
    michelson_contrast = (max_val - min_val) / (max_val + min_val + 1e-6)
    
    # RMS contrast
    rms_contrast = np.std(gray) / np.mean(gray)
    
    features = {
        'michelson_contrast': michelson_contrast,
        'rms_contrast': rms_contrast,
        'local_contrast_3x3_mean': np.mean(np.abs(local_contrast_3x3)),
        'local_contrast_5x5_mean': np.mean(np.abs(local_contrast_5x5)),
        'local_contrast_3x3_std': np.std(local_contrast_3x3),
        'local_contrast_5x5_std': np.std(local_contrast_5x5),
    }
    
    return features

# Add this at the bottom of the file
__all__ = [
    'analyze_dataset',
    'plot_size_distribution',
    'plot_brightness_analysis',
    'plot_sample_color_distributions',
    'generate_summary_report',
    'load_or_process_dataset',
    'analyze_feature_differences',
    'plot_correlation_matrix',
    'perform_pca_analysis',
    'plot_feature_distributions',
    'generate_analysis_report',
    'analyze_features',
    'extract_line_features',
    'extract_contrast_features',
    'extract_color_features',
    'extract_texture_features',
    'extract_line_features',
    'extract_contrast_features',
    'load_or_process_dataset',
    'plot_size_distribution',
    'plot_brightness_analysis',
    'plot_sample_color_distributions',
    'generate_summary_report'
]