"""Classification-based anomaly detection using Isolation Forest, SVM, and LOF."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from scipy.ndimage import gaussian_filter1d
import warnings

from .base import BaseDetector

warnings.filterwarnings('ignore')


class ClassificationDetector(BaseDetector):
    """
    Classification-based anomaly detector.
    
    Uses an ensemble of:
    - Isolation Forest (multiple contamination values)
    - One-Class SVM (multiple nu values)
    - Local Outlier Factor (multiple neighbor counts)
    """
    
    def __init__(self, window_size: int = None):
        super().__init__(window_size)
    
    def detect(self, X: np.ndarray, test_start: int) -> tuple:
        """
        Detect anomalies using classification-based methods.
        
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
        
        # Ensemble of Classification-Based Methods
        all_scores = []
        
        # Method A: Isolation Forest with different contamination values
        for cont in [0.01, 0.05, 0.1]:
            try:
                iso_forest = IsolationForest(
                    contamination=cont, random_state=42, n_estimators=100
                )
                iso_forest.fit(X_train_scaled)
                iso_scores = -iso_forest.decision_function(X_all_scaled)
                all_scores.append(iso_scores)
            except:
                pass
        
        # Method B: One-Class SVM
        for nu in [0.01, 0.05, 0.1]:
            try:
                ocsvm = OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
                ocsvm.fit(X_train_scaled)
                svm_scores = -ocsvm.decision_function(X_all_scaled)
                all_scores.append(svm_scores)
            except:
                pass
        
        # Method C: LOF with novelty detection
        for n_neighbors in [10, 20, 30, 50]:
            try:
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors, novelty=True, contamination=0.01
                )
                lof.fit(X_train_scaled)
                lof_scores = -lof.decision_function(X_all_scaled)
                all_scores.append(lof_scores)
            except:
                pass
        
        # Normalize and combine scores
        normalized_scores = []
        for s in all_scores:
            s_norm = self._normalize_scores(s)
            normalized_scores.append(s_norm)
        
        combined = np.mean(normalized_scores, axis=0)
        
        # Smoothing
        smooth_window = max(20, window_size)
        score = gaussian_filter1d(combined, sigma=smooth_window/3)
        
        # Zero out training region
        score[:test_start] = 0
        
        # Note: Using argmin because classification methods can have inverted scores
        predicted = test_start + np.argmin(score[test_start:])
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
                window[-1] - window[0],  # Trend
                np.max(window) - np.min(window),  # Range
                np.percentile(window, 75) - np.percentile(window, 25),  # IQR
            ]
            # First derivative features
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
def classification_based_scores(X: np.ndarray, test_start: int) -> tuple:
    """Detect anomalies using classification-based methods."""
    detector = ClassificationDetector()
    return detector.detect(X, test_start)
