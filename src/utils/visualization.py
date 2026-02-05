"""Visualization utilities for anomaly detection results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_with_anomaly_score(
        data: np.ndarray, 
        score: np.ndarray, 
        test_start: int,
        predicted: int, 
        anomaly: tuple, 
        name: str = None,
        save_path: str = None
    ) -> None:
    """
    Visualize time series with anomaly scores and predictions.
    
    Args:
        data: Raw time series data
        score: Anomaly scores for each point
        test_start: Index where test segment begins
        predicted: Predicted anomaly location
        anomaly: Tuple (start, end) of actual anomaly, or (-1, -1)
        name: Optional title for the plot
        save_path: Optional path to save the figure
    """
    anomaly_start = anomaly[0]
    anomaly_end = anomaly[1]
    predicted_start = max(0, predicted - 50)
    predicted_end = min(predicted + 50, data.shape[-1])
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 4), sharex=True)
    
    # Plot test and train segments
    sns.lineplot(x=np.arange(test_start, len(data)), y=data[test_start:], 
                 ax=ax[0], lw=0.5, label="Test")
    sns.lineplot(x=np.arange(0, test_start), y=data[:test_start], 
                 ax=ax[0], lw=0.5, label="Train")
    
    # Plot actual anomaly if known
    if anomaly_start > 0:
        sns.lineplot(x=np.arange(anomaly_start, anomaly_end),
                     y=data[anomaly_start:anomaly_end], 
                     ax=ax[0], label="Actual", color="green")
    
    # Plot anomaly scores
    sns.lineplot(x=np.arange(len(score)), y=score, 
                 ax=ax[1], label="Anomaly Scores")
    
    # Plot predicted anomaly region
    sns.lineplot(x=np.arange(predicted_start, predicted_end),
                 y=data[predicted_start:predicted_end], 
                 ax=ax[0], label="Predicted", color="red")

    if name is not None:
        ax[0].set_title(name)
    
    sns.despine()
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
