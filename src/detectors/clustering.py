"""Clustering-based anomaly detection using KMeans, LOF, and kNN."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import warnings

from .base import BaseDetector

warnings.filterwarnings('ignore')


class ClusteringDetector(BaseDetector):
    """
    Clustering-based anomaly detector.
    
    Uses an ensemble of:
    - Local Outlier Factor (LOF)
    - KMeans cluster distances
    - k-Nearest Neighbors distances
    
    This is the best performing detector in the ensemble.
    """
    
    def __init__(self, window_size: int = None):
        super().__init__(window_size)
    
    def detect(self, X: np.ndarray, test_start: int) -> tuple:
        """
        Detect anomalies using clustering-based methods.
        
        Args:
            X: 2D array of sliding window features
            test_start: Index where test segment begins
            
        Returns:
            Tuple of (scores, predicted_location)
        """
        score = np.zeros(len(X))
        window_size = X.shape[1]
        
        # Feature Engineering
        X_features = self._extract_features(X)
        
        # Train/Test Split
        train_end = max(0, test_start - window_size + 1)
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_features[:train_end])
        X_all_scaled = scaler.transform(X_features)
        
        # Ensemble of Methods
        all_scores = []
        
        # Method A: LOF
        for n in [10, 30]:
            lof = LocalOutlierFactor(n_neighbors=n, novelty=True, contamination=0.001)
            lof.fit(X_train_scaled)
            all_scores.append(-lof.decision_function(X_all_scaled))
        
        # Method B: KMeans Distance
        for n in [5, 10]:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            kmeans.fit(X_train_scaled)
            distances = kmeans.transform(X_all_scaled)
            all_scores.append(np.min(distances, axis=1))
        
        # Method C: kNN Distance
        for k in [5, 15]:
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(X_train_scaled)
            dists, _ = knn.kneighbors(X_all_scaled)
            all_scores.append(np.mean(dists, axis=1))
        
        # Robust Combination
        normalized_scores = []
        score_scaler = MinMaxScaler()
        
        for s in all_scores:
            cap = np.percentile(s, 99)
            s_clipped = np.clip(s, a_min=None, a_max=cap)
            s_norm = score_scaler.fit_transform(s_clipped.reshape(-1, 1)).flatten()
            normalized_scores.append(s_norm)
        
        combined = np.mean(normalized_scores, axis=0)
        
        # Smoothing
        filter_sigma = max(5, window_size / 3)
        score = gaussian_filter1d(combined, sigma=filter_sigma)
        
        # Zero out training region
        score[:test_start] = 0
        
        predicted = test_start + np.argmax(score[test_start:])
        return score, predicted
    
    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract statistical features from sliding windows."""
        f_std = np.std(X, axis=1)
        f_trend = X[:, -1] - X[:, 0]
        f_range = np.ptp(X, axis=1)
        f_iqr = np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1)
        
        diffs = np.diff(X, axis=1)
        f_d_mean = np.mean(diffs, axis=1)
        f_d_std = np.std(diffs, axis=1)
        f_d_max = np.max(np.abs(diffs), axis=1)
        
        # Peak Location
        f_argmax = np.argmax(X, axis=1).astype(float)
        
        # Asymmetry
        mid_point = X.shape[1] // 2
        left_std = np.std(X[:, :mid_point], axis=1)
        right_std = np.std(X[:, mid_point:], axis=1)
        f_asymmetry = left_std - right_std
        
        # Center of Mass
        indices = np.arange(X.shape[1])
        X_pos = X - np.min(X, axis=1, keepdims=True) + 1e-5
        f_center_mass = np.sum(indices * X_pos, axis=1) / np.sum(X_pos, axis=1)
        
        return np.column_stack([
            f_std, f_trend, f_range, f_iqr,
            f_d_mean, f_d_std, f_d_max,
            f_argmax, f_asymmetry, f_center_mass
        ])


# Standalone function for backward compatibility
def cluster_based_scores(X: np.ndarray, test_start: int) -> tuple:
    """
    Detect anomalies using clustering-based methods.
    
    This is a standalone function for backward compatibility with the notebook.
    """
    detector = ClusteringDetector()
    return detector.detect(X, test_start)
