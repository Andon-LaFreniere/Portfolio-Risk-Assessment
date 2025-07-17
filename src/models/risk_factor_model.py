from typing import Any, Optional
import numpy as np
from sklearn.decomposition import PCA

class RiskFactorModel:
    """
    PCA/Factor analysis for systematic risk decomposition.
    """
    def __init__(self, n_factors: int = 3):
        self.n_factors = n_factors
        self.pca = PCA(n_components=n_factors)

    def fit(self, X: np.ndarray):
        """Fit the PCA model to the data."""
        try:
            self.pca.fit(X)
        except Exception as e:
            print(f"Error fitting PCA: {e}")
            raise

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using the fitted PCA model."""
        try:
            return self.pca.transform(X)
        except Exception as e:
            print(f"Error transforming with PCA: {e}")
            raise 