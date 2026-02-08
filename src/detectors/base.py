"""Base class for all anomaly detectors."""

from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for anomaly detection methods.
    
    All detector implementations should inherit from this class
    and implement the `detect` method.
    """
    
    def __init__(self, window_size: int = None):
        """
        Initialize the detector.
        
        Args:
            window_size: Size of the sliding window for feature extraction
        """
        self.window_size = window_size
    
    @abstractmethod
    def detect(self, X: np.ndarray, test_start: int) -> tuple:
        """
        Detect anomalies in the time series.
        
        Args:
            X: 2D array of sliding window features
            test_start: Index where test segment begins
            
        Returns:
            Tuple of (scores, predicted_location)
            - scores: Anomaly score for each point
            - predicted_location: Index of predicted anomaly center
        """
        pass
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val > 0:
            return (scores - min_val) / (max_val - min_val + 1e-10)
        return scores
    
