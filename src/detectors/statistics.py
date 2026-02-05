"""Statistics-based anomaly detection using KDE, Mahalanobis, Z-score, and IQR."""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import RobustScaler
import warnings

from .base import BaseDetector

warnings.filterwarnings('ignore')


class StatisticsDetector(BaseDetector):
    """
    Statistics-based anomaly detector.
    
    Uses an ensemble of:
    - Kernel Density Estimation (KDE)
    - Mahalanobis distance
    - Per-feature Z-score
    - IQR-based outlier detection
    """
    
    def __init__(self, window_size: int = None):
        super().__init__(window_size)
    
    def detect(self, X: np.ndarray, test_start: int, data: np.ndarray = None) -> tuple:
        """
        Detect anomalies using statistical methods.
        
        Args:
            X: 2D array of sliding window features
            test_start: Index where test segment begins
            data: Raw time series data (optional, for compatibility)
            
        Returns:
            Tuple of (scores, predicted_location)
        """
        window_size = X.shape[1]
        
        # Feature Engineering
        X_features = self._extract_features(X)
        
        # Scale based on training data
        scaler = RobustScaler()
        train_end = test_start - window_size + 1
        X_train_scaled = scaler.fit_transform(X_features[:train_end])
        X_all_scaled = scaler.transform(X_features)
        
        all_scores = []
        
        # Multi-dimensional KDE
        try:
            kde = gaussian_kde(X_train_scaled.T)
            likelihoods = kde(X_all_scaled.T)
            kde_score = 1 / (likelihoods + 1e-10)
            kde_score = self._normalize_scores(kde_score)
            all_scores.append(kde_score)
        except:
            pass
        
        # Mahalanobis-like distance
        try:
            train_mean = np.mean(X_train_scaled, axis=0)
            train_cov = np.cov(X_train_scaled.T)
            inv_cov = np.linalg.inv(train_cov + np.eye(train_cov.shape[0]) * 1e-6)
            
            mahal_scores = []
            for i in range(len(X_all_scaled)):
                diff = X_all_scaled[i] - train_mean
                mahal = np.sqrt(diff @ inv_cov @ diff.T)
                mahal_scores.append(mahal)
            mahal_scores = np.array(mahal_scores)
            mahal_scores = self._normalize_scores(mahal_scores)
            all_scores.append(mahal_scores)
        except:
            pass
        
        # Per-feature Z-score outlier detection
        z_combined = np.zeros(len(X_all_scaled))
        for col in range(X_all_scaled.shape[1]):
            train_mean_col = np.mean(X_train_scaled[:, col])
            train_std_col = np.std(X_train_scaled[:, col]) + 1e-10
            z_scores = np.abs((X_all_scaled[:, col] - train_mean_col) / train_std_col)
            z_combined += z_scores
        z_combined = self._normalize_scores(z_combined)
        all_scores.append(z_combined)
        
        # IQR-based multi-feature
        iqr_combined = np.zeros(len(X_all_scaled))
        for col in range(X_train_scaled.shape[1]):
            Q1 = np.percentile(X_train_scaled[:, col], 25)
            Q3 = np.percentile(X_train_scaled[:, col], 75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            for i in range(len(X_all_scaled)):
                if X_all_scaled[i, col] < lower:
                    iqr_combined[i] += lower - X_all_scaled[i, col]
                elif X_all_scaled[i, col] > upper:
                    iqr_combined[i] += X_all_scaled[i, col] - upper
        if np.max(iqr_combined) > 0:
            iqr_combined = iqr_combined / np.max(iqr_combined)
        all_scores.append(iqr_combined)
        
        # Combine all scores
        combined = np.mean(all_scores, axis=0)
        
        # Smoothing
        smooth_window = max(20, window_size)
        score = gaussian_filter1d(combined, sigma=smooth_window/3)
        
        # Zero training region
        score[:test_start] = 0
        
        predicted = test_start + np.argmax(score[test_start:])
        return score, predicted
    
    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract statistical features from sliding windows."""
        features = []
        for i in range(len(X)):
            window = X[i]
            feat = [
                np.mean(window),
                np.std(window),
                window[-1],
                window[-1] - window[0],
                np.max(window) - np.min(window),
                np.percentile(window, 75) - np.percentile(window, 25),
            ]
            diffs = np.diff(window)
            feat.extend([np.mean(diffs), np.std(diffs), np.max(np.abs(diffs))])
            features.append(feat)
        return np.array(features)


# Standalone function for backward compatibility
def statistics_based_scores(data: np.ndarray, X: np.ndarray, test_start: int) -> tuple:
    """Detect anomalies using statistical methods."""
    detector = StatisticsDetector()
    return detector.detect(X, test_start, data)
