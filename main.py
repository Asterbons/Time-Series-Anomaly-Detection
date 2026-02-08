"""
Time Series Anomaly Detection - Command Line Interface

This script runs anomaly detection on time series data using various methods.

Usage:
    python main.py --method clustering
    python main.py --method all --save-plots
"""

import argparse
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from aeon.segmentation import find_dominant_window_sizes

from src.utils.data_loader import load_dataset, read_series, sliding_window
from src.utils.visualization import visualize_with_anomaly_score
from src.detectors import (
    ClusteringDetector,
    ClassificationDetector,
    RegressionDetector,
    ForecastingDetector,
    StatisticsDetector,
    NearestNeighborDetector
)


DETECTORS = {
    'clustering': ClusteringDetector,
    'classification': ClassificationDetector,
    'regression': RegressionDetector,
    'forecasting': ForecastingDetector,
    'statistics': StatisticsDetector,
    'nearest_neighbor': NearestNeighborDetector,
}


def run_detection(method: str, data_dir: str = ".", 
                  output_dir: str = "results",
                  save_plots: bool = True,
                  show_plots: bool = False) -> pd.DataFrame:
    """
    Run anomaly detection using the specified method.
    
    Args:
        method: Detection method name (or 'all' for ensemble)
        data_dir: Directory containing phase_1.zip and labels.csv
        output_dir: Directory to save results, scores, and plots
        save_plots: Whether to save visualization plots
        show_plots: Whether to display plots interactively
        
    Returns:
        DataFrame with predictions
    """
    import matplotlib
    if not show_plots:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    zip_path = os.path.join(data_dir, "phase_1.zip")
    labels_path = os.path.join(data_dir, "labels.csv")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{method}_{timestamp}")
    plots_dir = os.path.join(run_dir, "plots")
    scores_dir = os.path.join(run_dir, "scores")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)
    
    print(f"\nOutput directory: {run_dir}")
    
    # Load dataset
    file_list, locations, zf, folder_in_zip = load_dataset(zip_path, labels_path)
    
    # Select detector(s)
    if method == 'all':
        detector_classes = list(DETECTORS.values())
    else:
        if method not in DETECTORS:
            raise ValueError(f"Unknown method: {method}. Available: {list(DETECTORS.keys())}")
        detector_classes = [DETECTORS[method]]
    
    total_score = 0
    predictions = []
    ids = []
    all_results = []
    i = 1
    
    print(f"\n{'='*60}")
    print(f"Running anomaly detection with method: {method}")
    print(f"{'='*60}\n")
    
    for file in file_list:
        if "Anomaly" not in str(file):
            continue
            
        file_name = file.split('.')[0]
        name, test_start, data, anomaly = read_series(file, locations, zf, folder_in_zip)
        
        # Find dominant window size
        period = find_dominant_window_sizes(data[:test_start])
        window_size = int(period)
        
        print(f"Processing: {name} (window_size={window_size})")
        
        # Create sliding windows
        X = sliding_window(data, window_size)
        
        # Run detector(s)
        if len(detector_classes) == 1:
            detector = detector_classes[0](window_size=window_size)
            if method == 'statistics':
                score, predicted = detector.detect(X, test_start, data)
            else:
                score, predicted = detector.detect(X, test_start)
        else:
            # Ensemble: average predictions from all detectors
            all_predictions = []
            all_scores = []
            for DetectorClass in detector_classes:
                detector = DetectorClass(window_size=window_size)
                try:
                    if DetectorClass == StatisticsDetector:
                        s, p = detector.detect(X, test_start, data)
                    else:
                        s, p = detector.detect(X, test_start)
                    all_predictions.append(p)
                    all_scores.append(s)
                except Exception as e:
                    print(f"  Warning: {DetectorClass.__name__} failed: {e}")
            
            predicted = int(np.median(all_predictions))
            score = np.mean(all_scores, axis=0)
        
        predictions.append(predicted)
        ids.append(file_name)
        
        # Calculate error if anomaly is known
        error = None
        anomaly_center = None
        if anomaly[0] > -1:
            anomaly_center = (anomaly[0] + anomaly[1]) // 2
            error = abs(anomaly_center - predicted)
            total_score += error / anomaly_center
            i += 1
            print(f"  Predicted: {predicted}, Actual center: {anomaly_center}, Error: {error}")
        
        # Store detailed results
        all_results.append({
            'file_name': file_name,
            'test_start': test_start,
            'window_size': window_size,
            'predicted': predicted,
            'anomaly_start': anomaly[0] if anomaly[0] > -1 else None,
            'anomaly_end': anomaly[1] if anomaly[1] > -1 else None,
            'anomaly_center': anomaly_center,
            'error': error
        })
        
        # Save scores to CSV
        score_df = pd.DataFrame({
            'index': np.arange(len(score)),
            'score': score
        })
        score_file = os.path.join(scores_dir, f"{file_name}_scores.csv")
        score_df.to_csv(score_file, index=False)
        
        # Save plot
        if save_plots or show_plots:
            score_viz = score.copy()
            score_viz[:test_start] = np.nan
            
            # Create the plot
            import seaborn as sns
            
            anomaly_start = anomaly[0]
            anomaly_end = anomaly[1]
            predicted_start = max(0, predicted - 50)
            predicted_end = min(predicted + 50, data.shape[-1])
            
            fig, ax = plt.subplots(2, 1, figsize=(20, 6), sharex=True)
            
            # Plot time series
            ax[0].plot(np.arange(test_start, len(data)), data[test_start:], 
                      lw=0.5, label="Test", alpha=0.8)
            ax[0].plot(np.arange(0, test_start), data[:test_start], 
                      lw=0.5, label="Train", alpha=0.8)
            
            if anomaly_start > 0:
                ax[0].plot(np.arange(anomaly_start, anomaly_end),
                          data[anomaly_start:anomaly_end], 
                          lw=2, label="Actual Anomaly", color="green")
            
            ax[0].plot(np.arange(predicted_start, predicted_end),
                      data[predicted_start:predicted_end], 
                      lw=2, label="Predicted", color="red")
            
            ax[0].set_title(f"{name} - Method: {method}")
            ax[0].legend(loc='upper right')
            ax[0].set_ylabel("Value")
            
            # Plot anomaly scores
            ax[1].plot(np.arange(len(score_viz)), score_viz, lw=0.8, label="Anomaly Score")
            ax[1].axvline(x=predicted, color='red', linestyle='--', label=f'Predicted: {predicted}')
            if anomaly_center:
                ax[1].axvline(x=anomaly_center, color='green', linestyle='--', label=f'Actual: {anomaly_center}')
            ax[1].set_xlabel("Time Index")
            ax[1].set_ylabel("Anomaly Score")
            ax[1].legend(loc='upper right')
            
            plt.tight_layout()
            
            if save_plots:
                plot_file = os.path.join(plots_dir, f"{file_name}.png")
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    # Calculate final score
    final_score = (total_score / len(locations)) * 100
    
    print(f"\n{'='*60}")
    print(f"Final Score: {final_score:.2f}")
    print(f"{'='*60}\n")
    
    # Save predictions
    submission = pd.DataFrame({'ID': ids, 'PREDICTED': predictions})
    predictions_file = os.path.join(run_dir, "predictions.csv")
    submission.to_csv(predictions_file, index=False)
    print(f"Saved predictions to: {predictions_file}")
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(run_dir, "detailed_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Saved detailed results to: {results_file}")
    
    # Save summary
    summary = {
        'method': method,
        'timestamp': timestamp,
        'final_score': final_score,
        'num_files': len(ids),
        'mean_error': results_df['error'].mean() if results_df['error'].notna().any() else None
    }
    summary_file = os.path.join(run_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Anomaly Detection Results\n")
        f.write(f"{'='*40}\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved summary to: {summary_file}")
    
    print(f"\nAll outputs saved to: {run_dir}/")
    print(f"  - predictions.csv")
    print(f"  - detailed_results.csv")
    print(f"  - summary.txt")
    print(f"  - scores/ ({len(ids)} files)")
    print(f"  - plots/ ({len(ids)} files)")
    
    return submission


def main():
    parser = argparse.ArgumentParser(
        description='Time Series Anomaly Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --method clustering
  python main.py --method all --show-plots
  python main.py --method regression --no-save-plots
  
Available methods:
  clustering      - LOF, KMeans, kNN ensemble (BEST)
  classification  - Isolation Forest, SVM, LOF
  regression      - Gradient Boosting residuals
  forecasting     - Ridge regression + LOF
  statistics      - KDE, Mahalanobis, Z-score, IQR
  nearest_neighbor - kNN distances
  all             - Ensemble of all methods
  
Outputs (saved to results/<method>_<timestamp>/):
  - predictions.csv      : Predicted anomaly locations
  - detailed_results.csv : Full results with errors
  - summary.txt          : Run summary and final score
  - scores/              : Anomaly scores for each file
  - plots/               : Visualization plots
        """
    )
    
    parser.add_argument(
        '--method', '-m',
        type=str,
        default='clustering',
        choices=list(DETECTORS.keys()) + ['all'],
        help='Detection method to use (default: clustering)'
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='notebooks',
        help='Directory containing phase_1.zip and labels.csv (default: notebooks)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help='Directory to save outputs (default: results)'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        default=True,
        help='Save visualization plots (default: True)'
    )
    
    parser.add_argument(
        '--no-save-plots',
        action='store_true',
        help='Do not save visualization plots'
    )
    
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )
    
    args = parser.parse_args()
    
    save_plots = args.save_plots and not args.no_save_plots
    
    try:
        if args.method == 'all':
            all_submissions = {}
            for method_name in DETECTORS.keys():
                print(f"\n{'#'*60}")
                print(f"# Running individual method: {method_name}")
                print(f"{'#'*60}")
                submission = run_detection(
                    method=method_name,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    save_plots=save_plots,
                    show_plots=args.show_plots
                )
                all_submissions[method_name] = submission
            
            print(f"\n{'='*60}")
            print(f"All {len(DETECTORS)} methods completed!")
            print(f"{'='*60}")
            for method_name in all_submissions:
                print(f"  - {method_name}")
        else:
            submission = run_detection(
                method=args.method,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                save_plots=save_plots,
                show_plots=args.show_plots
            )
            print("\nPredictions preview:")
            print(submission.head(10).to_string(index=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
