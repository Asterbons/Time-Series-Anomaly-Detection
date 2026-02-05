"""Forecasting-based anomaly detection using prediction residuals and LOF."""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from scipy.ndimage import gaussian_filter1d
import warnings

from .base import BaseDetector

warnings.filterwarnings('ignore')


class ForecastingDetector(BaseDetector):
    """
    Forecasting-based anomaly detector.
    
    Uses an ensemble of:
    - Prediction residuals with multiple lag values
    - LOF on temporal context
    - kNN distance measures
    """
    
    def __init__(self, window_size: int = None):
        super().__init__(window_size)
    
    def detect(self, X: np.ndarray, test_start: int) -> tuple:
        """
        Detect anomalies using forecasting-based methods.
        
        Args:
            X: 2D array of sliding window features
            test_start: Index where test segment begins
            
        Returns:
            Tuple of (scores, predicted_location)
        """
        window_size = X.shape[1]
        
        # Feature Engineering
        X_features = self._extract_features(X)
        
        # Scaling
        scaler = RobustScaler()
        train_end = test_start - window_size + 1
        X_train_scaled = scaler.fit_transform(X_features[:train_end])
        X_all_scaled = scaler.transform(X_features)
        
        all_scores = []
        
        # Prediction residuals ensemble
        for lag in [1, 2, 3, 5]:
            if test_start > lag + 10:
                X_in = X_train_scaled[:-lag]
                y_out = X_train_scaled[lag:]
                
                model = Ridge(alpha=1.0)
                model.fit(X_in, y_out)
                
                pred_scores = np.zeros(len(X_all_scaled))
                for i in range(len(X_all_scaled) - lag):
                    pred = model.predict(X_all_scaled[i:i+1])
                    actual = X_all_scaled[i + lag]
                    pred_scores[i + lag] = np.linalg.norm(actual - pred)
                
                if np.max(pred_scores) > 0:
                    pred_scores = pred_scores / np.max(pred_scores)
                all_scores.append(pred_scores)
        
        # LOF on temporal context
        for n_neighbors in [10, 20, 30]:
            try:
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors, novelty=True, contamination=0.01
                )
                lof.fit(X_train_scaled)
                lof_scores = -lof.decision_function(X_all_scaled)
                lof_scores = self._normalize_scores(lof_scores)
                all_scores.append(lof_scores)
            except:
                pass
        
        # kNN distance
        for k in [5, 10, 15]:
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(X_train_scaled)
            distances, _ = knn.kneighbors(X_all_scaled)
            avg_dist = np.mean(distances, axis=1)
            avg_dist = self._normalize_scores(avg_dist)
            all_scores.append(avg_dist)
        
        # Combine
        combined = np.mean(all_scores, axis=0)
        
        # Smoothing
        smooth_window = max(20, window_size)
        score = gaussian_filter1d(combined, sigma=smooth_window/3)
        
        # Zero training
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
            
            # Second derivative
            diffs2 = np.diff(diffs)
            if len(diffs2) > 0:
                feat.extend([np.mean(diffs2), np.std(diffs2)])
            else:
                feat.extend([0, 0])
            features.append(feat)
        return np.array(features)


# Standalone function for backward compatibility
def forecasting_based_scores(X: np.ndarray, test_start: int) -> tuple:
    """Detect anomalies using forecasting-based methods."""
    detector = ForecastingDetector()
    return detector.detect(X, test_start)
