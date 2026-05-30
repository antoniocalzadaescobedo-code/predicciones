"""
================================================================================
  calibration.py — Modular Multi-class Probability Calibration Layer
  FIFA World Cup 2026 Predictor
================================================================================
  This module provides a scientifically correct probability calibrator for
  3-way classification outcomes (Home Win, Draw, Away Win).
  
  Implemented Calibrators:
    1. TemperatureScalingCalibrator (Only active calibrator in this phase)
    
  Placeholders for future work:
    - DirichletCalibrator
    - PlattMulticlassCalibrator
    - IsotonicMulticlassCalibrator
================================================================================
"""

import os
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
from scipy.optimize import minimize

EPSILON = 1e-12

def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Converts a probability matrix to logits (log-odds) with clipping to prevent log(0)."""
    clipped = np.clip(probs, EPSILON, 1.0 - EPSILON)
    return np.log(clipped)

def softmax(x: np.ndarray) -> np.ndarray:
    """Computes row-wise softmax in a numerically stable way."""
    max_x = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class BaseCalibrator(ABC):
    """Abstract Base Class for multi-class probability calibrators."""
    
    def __init__(self):
        self.is_fitted = False
        self.fit_metadata = {}

    @abstractmethod
    def fit(self, probs: np.ndarray, actuals: np.ndarray) -> "BaseCalibrator":
        """
        Fits the calibrator parameters using calibration data.
        
        Parameters
        ----------
        probs : np.ndarray of shape (N, 3)
            Raw predicted probabilities [p_home, p_draw, p_away]
        actuals : np.ndarray of shape (N,) or (N, 3)
            Actual outcomes. If 1D: values are 1 (home win), 0 (draw), -1 (away win).
            If 2D: one-hot encoded targets [home, draw, away].
        """
        pass

    @abstractmethod
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrates raw probabilities.
        
        Parameters
        ----------
        probs : np.ndarray of shape (N, 3)
            Raw predicted probabilities [p_home, p_draw, p_away]
            
        Returns
        -------
        calibrated_probs : np.ndarray of shape (N, 3)
            Calibrated probabilities that sum to 1.0.
        """
        pass

    def _prepare_targets(self, actuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper to convert actual outcomes into 1D class indices (0=home, 1=draw, 2=away)
        and 2D one-hot encodings.
        """
        if actuals.ndim == 1:
            # Map actual outcomes (1 -> 0, 0 -> 1, -1 -> 2)
            y_indices = np.zeros(actuals.shape[0], dtype=int)
            y_indices[actuals == 1] = 0
            y_indices[actuals == 0] = 1
            y_indices[actuals == -1] = 2
            
            y_onehot = np.zeros((actuals.shape[0], 3))
            y_onehot[np.arange(actuals.shape[0]), y_indices] = 1.0
        else:
            y_onehot = actuals.astype(float)
            y_indices = np.argmax(actuals, axis=1)
            
        return y_indices, y_onehot

    def save(self, filepath: str) -> None:
        """Persists the calibrator model to disk using Pickle."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath: str) -> "BaseCalibrator":
        """Loads a persisted calibrator model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class TemperatureScalingCalibrator(BaseCalibrator):
    """
    Temperature Scaling Calibrator.
    
    Applies a single scalar temperature parameter T to logits:
        calibrated_probs = softmax(logits / T)
        
    It has only 1 parameter (T > 0), making overfitting virtually impossible. 
    Preserves relative ranking of classes while adjusting global confidence.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = 1.0

    def fit(self, probs: np.ndarray, actuals: np.ndarray) -> "TemperatureScalingCalibrator":
        probs = np.asarray(probs, dtype=float)
        _, y_onehot = self._prepare_targets(actuals)
        
        # Convert raw probabilities to logits
        logits = probs_to_logits(probs)
        
        # Loss function: Negative Log-Likelihood (multiclass cross-entropy)
        def nll_loss(t_val: float) -> float:
            t_val = max(t_val, 0.01)  # prevent division by zero or negative T
            cal_probs = softmax(logits / t_val)
            clipped_cal_probs = np.clip(cal_probs, EPSILON, 1.0 - EPSILON)
            return -np.mean(np.sum(y_onehot * np.log(clipped_cal_probs), axis=1))

        # Optimize temperature scalar starting from T=1.0. Bounds: (0.05, 10.0)
        res = minimize(lambda x: nll_loss(x[0]), x0=[1.0], method="Nelder-Mead", bounds=[(0.05, 10.0)])
        self.temperature = float(res.x[0])
        self.is_fitted = True
        self.fit_metadata = {
            "temperature": self.temperature,
            "success": res.success,
            "final_loss": res.fun,
            "n_samples": probs.shape[0]
        }
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Calibrator is not fitted yet.")
        probs = np.asarray(probs, dtype=float)
        logits = probs_to_logits(probs)
        calibrated = softmax(logits / self.temperature)
        
        # Validation checks
        assert not np.isnan(calibrated).any(), "Calibrated probabilities contain NaN!"
        sums = np.sum(calibrated, axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5), "Calibrated probabilities do not sum to 1.0!"
        
        return calibrated


# ==============================================================================
# PLACEHOLDERS FOR FUTURE WORK (DEFERRED TO PREVENT OVERENGINEERING)
# ==============================================================================

class PlattMulticlassCalibrator(BaseCalibrator):
    """Placeholder for Platt Multiclass (logistic scaling) calibrator."""
    def fit(self, probs: np.ndarray, actuals: np.ndarray) -> "PlattMulticlassCalibrator":
        raise NotImplementedError("PlattMulticlassCalibrator is deferred to a future phase.")
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("PlattMulticlassCalibrator is deferred to a future phase.")


class DirichletCalibrator(BaseCalibrator):
    """Placeholder for Dirichlet calibrator."""
    def fit(self, probs: np.ndarray, actuals: np.ndarray) -> "DirichletCalibrator":
        raise NotImplementedError("DirichletCalibrator is deferred to a future phase.")
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("DirichletCalibrator is deferred to a future phase.")


class IsotonicMulticlassCalibrator(BaseCalibrator):
    """Placeholder for Isotonic regression calibrator."""
    def fit(self, probs: np.ndarray, actuals: np.ndarray) -> "IsotonicMulticlassCalibrator":
        raise NotImplementedError("IsotonicMulticlassCalibrator is deferred to a future phase.")
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("IsotonicMulticlassCalibrator is deferred to a future phase.")
