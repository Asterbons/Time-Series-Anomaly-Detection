"""
Compare Anomaly Detection Methods

This script reads results from all method folders and creates comparison visualizations.

Usage:
    python compare_methods.py
    python compare_methods.py --results-dir results
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def parse_summary(summary_path: str) -> dict:
    """Parse a summary.txt file and return metrics as a dictionary."""
    metrics = {}
    with open(summary_path, 'r') as f:
        for line in f:
            if ': ' in line and not line.startswith('='):
                key, value = line.strip().split(': ', 1)
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
    return metrics


def find_latest_results(results_dir: str) -> dict:
    """Find the most recent result folder for each method."""
    method_results = defaultdict(list)
    
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            summary_path = os.path.join(folder_path, 'summary.txt')
            if os.path.exists(summary_path):
                # Parse method name from folder (e.g., "clustering_20260207_200607")
                parts = folder.rsplit('_', 2)
                if len(parts) >= 3:
                    method_name = parts[0]
                    timestamp = f"{parts[1]}_{parts[2]}"
                    method_results[method_name].append((timestamp, folder_path))
    
    # Get the latest result for each method
    latest = {}
    for method, results in method_results.items():
        results.sort(key=lambda x: x[0], reverse=True)
        latest[method] = results[0][1]
    
    return latest


def create_comparison_chart(results_dir: str = "results", output_file: str = None):
    
    latest_results = find_latest_results(results_dir)
    
    if not latest_results:
        print("No results found!")
        return
    
    # Parse all summaries
    methods = []
    scores = []
    
    for method, folder_path in sorted(latest_results.items()):
        summary_path = os.path.join(folder_path, 'summary.txt')
        metrics = parse_summary(summary_path)
        
        methods.append(method.replace('_', ' ').title())
        scores.append(metrics.get('final_score', 0))
        
        print(f"{method}: score={metrics.get('final_score', 'N/A'):.2f}")
    
    # Sort by score 
    sorted_indices = np.argsort(scores)
    methods = [methods[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    methods = methods[::-1]
    scores = scores[::-1]
    
    # Create color gradient from green (best) to red (worst)
    n = len(methods)
    colors = []
    for i in range(n):
        ratio = i / (n - 1) if n > 1 else 1
        if ratio < 0.33:
            t = ratio / 0.33
            r, g, b = 1, 0.5 * t, 0
        elif ratio < 0.66:
            t = (ratio - 0.33) / 0.33
            r, g, b = 1, 0.5 + 0.5 * t, 0
        else:
            t = (ratio - 0.66) / 0.34
            r, g, b = 1 - t, 0.8 + 0.2 * (1 - t), 0
        colors.append((r, g, b))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(methods) * 0.8)))
    
    y_positions = np.arange(len(methods))
    bars = ax.barh(y_positions, scores, color=colors, edgecolor='black', linewidth=1.5, height=0.7)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods, fontsize=12, fontweight='bold')
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.annotate(f'{score:.1f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=12, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Final Score (lower is better)', fontsize=14, fontweight='bold')
    ax.set_title('Method Comparison by Score', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, max(scores) * 1.15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', labelsize=11)
    
    plt.tight_layout()
    
    if output_file is None:
        output_file = os.path.join(results_dir, 'methods_comparison.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nComparison chart saved to: {output_file}")
    
    plt.show()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    best_method = methods[-1]  # Best is now at top (last in reversed list)
    print(f"Best Method: {best_method} (Score: {scores[-1]:.2f})")


def main():
    parser = argparse.ArgumentParser(description='Compare anomaly detection methods')
    parser.add_argument('--results-dir', '-r', type=str, default='results',
                        help='Directory containing method results (default: results)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path for the chart')
    
    args = parser.parse_args()
    
    create_comparison_chart(args.results_dir, args.output)


if __name__ == '__main__':
    main()
