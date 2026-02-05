"""Regression-based anomaly detection using Gradient Boosting."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d

from .base import BaseDetector


class RegressionDetector(BaseDetector):
    """
    Regression-based anomaly detector.
    
    Uses Gradient Boosting regression to predict expected values
    and flags large residuals as anomalies.
    """
    
    def __init__(self, window_size: int = None, n_lags: int = 3):
        super().__init__(window_size)
        self.n_lags = n_lags
    
    def detect(self, X: np.ndarray, test_start: int) -> tuple:
        """
        Detect anomalies using regression-based prediction residuals.
        
        Args:
            X: 2D array of sliding window features
            test_start: Index where test segment begins
            
        Returns:
            Tuple of (scores, predicted_location)
        """
        score = np.zeros(len(X))
        n_lags = self.n_lags
        
        # Preprocessing: Z-score normalize
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X[:test_start])
        X_all = scaler.transform(X)
        
        # Use multiple lagged values as features
        X_train = X_all[:test_start-n_lags]
        y_train = X_all[n_lags:test_start, -1]
        
        # Use Gradient Boosting for better predictions
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predict for all data
        y_pred = model.predict(X_all[:-n_lags])
        y_actual = X_all[n_lags:, -1]
        
        # Use absolute residuals (more robust)
        residuals = np.abs(y_actual - y_pred)
        
        # Apply smoothing to reduce noise
        window_smooth = max(5, X.shape[1] // 4)
        residuals_smooth = uniform_filter1d(residuals, size=window_smooth)
        
        score[n_lags:len(residuals_smooth)+n_lags] = residuals_smooth
        
        predicted = test_start + np.argmax(score[test_start:])
        return score, predicted


# Standalone function for backward compatibility
def regression_based_scores(X: np.ndarray, test_start: int) -> tuple:
    """Detect anomalies using regression-based prediction residuals."""
    detector = RegressionDetector()
    return detector.detect(X, test_start)
