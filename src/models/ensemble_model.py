from typing import Any, Optional
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

class EnsembleModel:
    """
    Stacking/blending multiple models with uncertainty quantification.
    """
    def __init__(self, base_models: Optional[list] = None):
        self.base_models = base_models or []
        self.meta_model = LinearRegression()
        self.ensemble = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the ensemble model."""
        try:
            estimators = [(f"model_{i}", m) for i, m in enumerate(self.base_models)]
            self.ensemble = StackingRegressor(estimators=estimators, final_estimator=self.meta_model)
            self.ensemble.fit(X, y)
        except Exception as e:
            print(f"Error fitting ensemble: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the ensemble model."""
        if self.ensemble is None:
            raise ValueError("Ensemble not fitted.")
        try:
            return self.ensemble.predict(X)
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            raise 