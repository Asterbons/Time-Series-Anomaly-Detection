"""Nearest neighbor-based anomaly detection using kNN distances."""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseDetector


class NearestNeighborDetector(BaseDetector):
    """
    Nearest Neighbor-based anomaly detector.
    
    Uses k-Nearest Neighbors to compute distance-based anomaly scores.
    Points far from their neighbors in the training set are flagged as anomalies.
    """
    
    def __init__(self, window_size: int = None, n_neighbors: int = 5):
        super().__init__(window_size)
        self.n_neighbors = n_neighbors
    
    def detect(self, X: np.ndarray, test_start: int) -> tuple:
        """
        Detect anomalies using k-Nearest Neighbor distances.
        
        Args:
            X: 2D array of sliding window features
            test_start: Index where test segment begins
            
        Returns:
            Tuple of (scores, predicted_location)
        """
        score = np.zeros(len(X))
        window_size = X.shape[1]
        train_end = test_start - window_size + 1        
        knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        knn.fit(X[:train_end])
        
        # For each point, get distance to nearest neighbors in training set
        distances, _ = knn.kneighbors(X)
        
        # Average distance to k neighbors = anomaly score
        score = np.mean(distances, axis=1)
        
        predicted = test_start + np.argmax(score[test_start:])
        return score, predicted


# Standalone function for backward compatibility
def nearest_neighbor_based_score(X: np.ndarray, test_start: int, 
                                  window_size: int = None) -> tuple:
    """Detect anomalies using k-Nearest Neighbor distances."""
    detector = NearestNeighborDetector(window_size=window_size)
    return detector.detect(X, test_start)
