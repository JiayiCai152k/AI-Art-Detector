import pandas as pd
import numpy as np
from data_preprocessing.analyze import (
    load_or_process_dataset,
    analyze_feature_differences,
    plot_correlation_matrix,
    perform_pca_analysis,
    plot_feature_distributions,
    generate_analysis_report
)

def print_detailed_analysis():
    print("Loading and analyzing dataset...")
    df = load_or_process_dataset(force_reprocess=True)
    
    # Analyze feature differences
    feature_analysis = analyze_feature_differences(df)
    
    print("\n=== Top Differentiating Features Between AI and Human Artworks ===")
    print("\nTop 10 Most Significant Features:")
    top_features = feature_analysis.head(10)
    for _, row in top_features.iterrows():
        print(f"\nFeature: {row['feature']}")
        print(f"Effect Size: {row['effect_size']:.3f}")
        print(f"AI Mean: {row['ai_mean']:.3f}")
        print(f"Human Mean: {row['human_mean']:.3f}")
        print(f"Absolute Difference: {row['difference']:.3f}")
        print(f"P-value: {row['p_value']:.6f}")
    
    # Generate visualizations
    print("\nGenerating visualization plots...")
    plot_correlation_matrix(df)
    pca, _ = perform_pca_analysis(df)
    plot_feature_distributions(df, feature_analysis)
    
    # Generate comprehensive report
    print("\n=== Comprehensive Analysis Report ===")
    report = generate_analysis_report(df, feature_analysis, pca)
    print(report)
    
    return df, feature_analysis

if __name__ == "__main__":
    df, feature_analysis = print_detailed_analysis() 