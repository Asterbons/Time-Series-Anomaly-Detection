"""
Generate comparison chart of detector methods.

Usage:
    python scripts/compare_methods.py              # Generate chart
    python scripts/compare_methods.py --run-all    # Run all methods first, then generate chart
"""

import sys
import os
import argparse
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from aeon.segmentation import find_dominant_window_sizes
from src.utils.data_loader import load_dataset, read_series, sliding_window
from src.detectors import (
    ClusteringDetector,
    ClassificationDetector,
    RegressionDetector,
    ForecastingDetector,
    StatisticsDetector,
    NearestNeighborDetector
)

DETECTORS = {
    'Clustering': ClusteringDetector,
    'Classification': ClassificationDetector,
    'Regression': RegressionDetector,
    'Forecasting': ForecastingDetector,
    'Statistics': StatisticsDetector,
    'Nearest Neighbor': NearestNeighborDetector,
}


def run_all_methods():
    """Run all detection methods and return scores."""
    print("Loading data...")
    file_list, locations, zf, folder = load_dataset('notebooks/phase_1.zip', 'notebooks/labels.csv')
    
    results = {name: [] for name in DETECTORS.keys()}
    
    for file in file_list:
        if "Anomaly" not in str(file):
            continue
        
        name, test_start, data, anomaly = read_series(file, locations, zf, folder)
        
        # Get window size
        period = find_dominant_window_sizes(data[:test_start])
        window_size = int(period)
        X = sliding_window(data, window_size)
        
        anomaly_center = (anomaly[0] + anomaly[1]) // 2
        
        print(f"Processing: {name}")
        
        for method_name, DetectorClass in DETECTORS.items():
            try:
                detector = DetectorClass(window_size=window_size)
                if DetectorClass == StatisticsDetector:
                    _, predicted = detector.detect(X, test_start, data)
                else:
                    _, predicted = detector.detect(X, test_start)
                
                error = abs(anomaly_center - predicted) / anomaly_center
                results[method_name].append(error)
            except Exception as e:
                print(f"  Warning: {method_name} failed: {e}")
                results[method_name].append(1.0)  # Penalty for failure
    
    # Calculate final scores
    scores = {}
    for method_name, errors in results.items():
        scores[method_name] = sum(errors) / len(errors) * 100
    
    return scores


def generate_comparison_chart(scores: dict, output_path: str = 'assets/method_comparison.png'):
    """Generate bar chart comparing method scores."""
    
    # Sort by score (lower is better)
    sorted_methods = sorted(scores.items(), key=lambda x: x[1])
    methods = [m[0] for m in sorted_methods]
    values = [m[1] for m in sorted_methods]
    
    # Colors - best is green, worst is red
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(methods, values, color=colors[:len(methods)], edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                f'{value:.2f}%', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Add "BEST" label to the top method
    ax.text(values[0] / 2, 0, 'BEST', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    
    ax.set_xlabel('Score (Lower is Better)', fontsize=12)
    ax.set_title('Anomaly Detection Method Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.2)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {output_path}")
    
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Compare detector methods')
    parser.add_argument('--results-dir', '-r', type=str, default='results',
                        help='Directory containing method results')
    parser.add_argument('--output', '-o', type=str, default='assets/method_comparison.png',
                        help='Output file path')
    
    args = parser.parse_args()
    
    # Run all detection methods using main.py
    print("Running all detection methods...")
    import subprocess
    subprocess.run([sys.executable, 'main.py', '--method', 'all'], check=True)
    
    # Visualize results using results_visualizer
    print("\nGenerating comparison visualization...")
    from scripts.results_visualizer import create_comparison_chart
    create_comparison_chart(args.results_dir, args.output)


if __name__ == '__main__':
    main()
