"""
Generate visualization for README or any dataset.

Usage:
    python scripts/generate_readme_image.py                    # Default: 003_Anomaly_4375
    python scripts/generate_readme_image.py --dataset 005      # Use dataset 005
    python scripts/generate_readme_image.py --list             # List all datasets
    python scripts/generate_readme_image.py --dataset 010 --output my_plot.png
"""

import sys
import os
import argparse
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from aeon.segmentation import find_dominant_window_sizes
from src.utils.data_loader import load_dataset, read_series, sliding_window
from src.detectors.clustering import ClusteringDetector


def generate_visualization(dataset_id: str, output_path: str = None, zoom_margin: int = 800):
    """Generate anomaly detection visualization for a dataset."""
    
    # Load data
    file_list, locations, zf, folder = load_dataset('notebooks/phase_1.zip', 'notebooks/labels.csv')
    
    # Find the target file
    target_file = None
    for f in file_list:
        if f.startswith(dataset_id) and 'Anomaly' in f:
            target_file = f
            break
    
    if target_file is None:
        print(f"Error: Dataset '{dataset_id}' not found.")
        print("Use --list to see available datasets.")
        return None
    
    name, test_start, data, anomaly = read_series(target_file, locations, zf, folder)
    
    print(f"Generating visualization for: {name}")
    print(f"Anomaly: {anomaly}")
    
    # Get window size and run detection
    period = find_dominant_window_sizes(data[:test_start])
    window_size = int(period)
    X = sliding_window(data, window_size)
    
    detector = ClusteringDetector()
    score, predicted = detector.detect(X, test_start)
    
    anomaly_center = (anomaly[0] + anomaly[1]) // 2
    error = abs(predicted - anomaly_center)
    print(f"Predicted: {predicted}, Actual center: {anomaly_center}, Error: {error}")
    
    # Define zoom window
    zoom_start = max(0, anomaly[0] - zoom_margin)
    zoom_end = min(len(data), anomaly[1] + zoom_margin)
    
    # Create figure
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    
    # Colors
    COLOR_TRAIN = '#3498db'      # Blue
    COLOR_TEST = '#e67e22'       # Orange
    COLOR_ANOMALY = '#e74c3c'    # Red
    COLOR_PREDICTED = '#2ecc71'  # Green
    
    # ---- Top plot: Time series (zoomed) ----
    x_range = np.arange(zoom_start, zoom_end)
    y_data = data[zoom_start:zoom_end]
    
    train_mask = x_range < test_start
    test_mask = x_range >= test_start
    
    # Plot test data (orange)
    ax[0].plot(x_range[test_mask], y_data[test_mask], 
              lw=1.2, label="Test Data", color=COLOR_TEST, alpha=0.9)
    
    # Plot train data (blue) if visible
    if np.any(train_mask):
        ax[0].plot(x_range[train_mask], y_data[train_mask], 
                  lw=1.2, label="Training Data", color=COLOR_TRAIN, alpha=0.9)
    
    # Predicted area (green shaded)
    predicted_start = max(zoom_start, predicted - 50)
    predicted_end = min(zoom_end, predicted + 50)
    ax[0].axvspan(predicted_start, predicted_end, alpha=0.25, color=COLOR_PREDICTED, 
                  label='Predicted Region')
    
    # Anomaly as red line overlay
    anomaly_x = np.arange(anomaly[0], anomaly[1])
    anomaly_y = data[anomaly[0]:anomaly[1]]
    ax[0].plot(anomaly_x, anomaly_y, lw=1.2, color=COLOR_ANOMALY, label='Actual Anomaly')
    
    ax[0].set_ylabel('Value', fontsize=12)
    ax[0].set_title(f'Anomaly Detection Result: {name} (Zoomed View)', fontsize=14, fontweight='bold')
    ax[0].legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax[0].grid(True, alpha=0.3, linestyle='--')
    ax[0].set_xlim(zoom_start, zoom_end)
    
    # ---- Bottom plot: Anomaly scores ----
    score_viz = score.copy()
    score_viz[:test_start] = np.nan
    
    score_x = np.arange(zoom_start, min(zoom_end, len(score_viz)))
    score_y = score_viz[zoom_start:min(zoom_end, len(score_viz))]
    
    ax[1].plot(score_x, score_y, lw=1.2, color=COLOR_TEST, label='Anomaly Score')
    ax[1].fill_between(score_x, 0, score_y, alpha=0.3, color=COLOR_TEST)
    
    ax[1].axvline(x=predicted, color=COLOR_PREDICTED, linestyle='-', lw=2.5, 
                  label=f'Predicted: {predicted}')
    ax[1].axvline(x=anomaly_center, color=COLOR_ANOMALY, linestyle='--', lw=2.5, 
                  label=f'Actual: {anomaly_center}')
    
    ax[1].set_xlabel('Time Index', fontsize=12)
    ax[1].set_ylabel('Anomaly Score', fontsize=12)
    ax[1].legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax[1].grid(True, alpha=0.3, linestyle='--')
    ax[1].set_xlim(zoom_start, zoom_end)
    
    # Add annotation
    if error == 0:
        ax[1].annotate('Perfect Match!', xy=(predicted, 0.95), fontsize=11, 
                       color=COLOR_PREDICTED, fontweight='bold', ha='center', va='top')
    elif error < 50:
        ax[1].annotate(f'Error: {error}', xy=(predicted, 0.95), fontsize=11, 
                       color=COLOR_PREDICTED, fontweight='bold', ha='center', va='top')
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = f'assets/{name}.png'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {output_path}")
    
    plt.close()
    return output_path


def list_datasets():
    """List all available datasets."""
    file_list, _, _, _ = load_dataset('notebooks/phase_1.zip', 'notebooks/labels.csv')
    
    print("\nAvailable datasets:")
    print("-" * 40)
    for f in sorted(file_list):
        if 'Anomaly' in f:
            dataset_id = f.split('_')[0]
            print(f"  {dataset_id}  ->  {f}")
    print("-" * 40)
    print(f"Total: {len([f for f in file_list if 'Anomaly' in f])} datasets")


def main():
    parser = argparse.ArgumentParser(
        description='Generate anomaly detection visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_readme_image.py                          # Default (003)
  python scripts/generate_readme_image.py --dataset 005
  python scripts/generate_readme_image.py --dataset 010 --output my_viz.png
  python scripts/generate_readme_image.py --list
        """
    )
    
    parser.add_argument('--dataset', '-d', type=str, default='003',
                        help='Dataset ID (e.g., 003, 005, 010). Default: 003')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: assets/<dataset_name>.png)')
    parser.add_argument('--zoom', '-z', type=int, default=800,
                        help='Zoom margin around anomaly (default: 800)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    generate_visualization(args.dataset, args.output, args.zoom)


if __name__ == '__main__':
    main()
