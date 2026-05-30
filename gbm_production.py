"""
GBM PRODUCTION - Predictor oficial para FIFA World Cup 2026
==========================================================

Backbone: Gradient Boosting Multiclase [-1, 0, 1]
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class FIFA2026Predictor:
    """
    Predictor oficial para FIFA World Cup 2026.
    Backbone: Gradient Boosting Multiclase [-1, 0, 1]
    
    Uso:
        predictor = FIFA2026Predictor(calibrate=False)
        predictor.fit(X_train, y_train)
        probs = predictor.predict_proba(X_test)  # shape (N, 3), orden: [away, draw, home]
        preds = predictor.predict(X_test)        # [-1, 0, 1]
    """
    
    def __init__(self, calibrate: bool = False, n_estimators: int = 300, max_depth: int = 5):
        self.calibrate = calibrate
        self.target_classes = np.array([-1, 0, 1])  # [away_win, draw, home_win]
        
        base_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        self.model = CalibratedClassifierCV(base_model, cv=3, method='isotonic') if calibrate else base_model
        self.is_fitted = False
        self.feature_names = None
        self.metrics_history = {}
    
    def fit(self, X, y, feature_names=None):
        """Entrena el modelo GBM + Calibración con nuevos datos."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        
        # 1. Modelo Base
        base = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # 2. Calibración
        if self.calibrate:
            self.model = CalibratedClassifierCV(base, cv=5, method='isotonic')
        else:
            self.model = base
        
        # 3. Manejo de feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns) if feature_names is None else feature_names
            X = X.values
        else:
            self.feature_names = feature_names
            
        # 4. Entrenamiento
        y = np.asarray(y)
        self.model.fit(X, y)
        self.is_fitted = True
        print("✅ Modelo entrenado exitosamente.")
        return self
    
    def predict_proba(self, X):
        """Retorna probabilidades en orden estricto: [away_win, draw, home_win]"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        proba_raw = self.model.predict_proba(X)
        return self._align_probabilities(proba_raw)
    
    def predict(self, X):
        """Retorna predicciones: -1 (away), 0 (draw), 1 (home)"""
        proba = self.predict_proba(X)
        return self.target_classes[np.argmax(proba, axis=1)]
    
    def predict_match(self, team_home, team_away, features, neutral=False):
        """Predicción para un partido individual"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        
        # Construir feature vector
        feature_vector = np.array([
            features.get('elo_diff', 0),
            int(neutral),
            features.get('form_home', 0.5),
            features.get('form_away', 0.5),
            features.get('h2h', 0.5)
        ]).reshape(1, -1)
        
        proba = self.predict_proba(feature_vector)[0]
        pred = self.predict(feature_vector)[0]
        
        # Mapeo de predicción a texto
        pred_map = {-1: 'away_win', 0: 'draw', 1: 'home_win'}
        
        return {
            'teams': {'home': team_home, 'away': team_away},
            'prediction': {
                'outcome': pred_map[pred],
                'confidence': float(proba[np.argmax(proba)])
            },
            'probabilities': {
                'away_win': float(proba[0]),
                'draw': float(proba[1]),
                'home_win': float(proba[2])
            }
        }
    
    def predict_friendly(self, home, away, teams_db, neutral=False, motivation=0.85):
        """Predicción para partidos amistosos con ajuste de incertidumbre"""
        elo_diff = teams_db.get_elo_diff(home, away, neutral)
        features = {
            "elo_diff": elo_diff,
            "form_home": 0.50,  # Fallback seguro para amistosos
            "form_away": 0.50,
            "h2h": 0.50,
            "is_neutral": 1.0 if neutral else 0.0
        }
        res = self.predict_match(home, away, features, neutral)
        # Suavizar por naturaleza de amistoso
        p = res["probabilities"]
        adj = {k: v*motivation + (1-motivation)/3 for k, v in p.items()}
        total = sum(adj.values())
        res["probabilities"] = {k: v/total for k, v in adj.items()}
        res["match_type"] = "friendly"
        return res
    
    def _align_probabilities(self, proba_raw):
        """Asegura orden [-1, 0, 1] independientemente del orden interno del modelo"""
        if hasattr(self.model, 'classes_'):
            model_classes = np.asarray(self.model.classes_)
        else:
            model_classes = self.target_classes
            
        if np.array_equal(model_classes, self.target_classes):
            aligned = proba_raw
        else:
            aligned = np.zeros((proba_raw.shape[0], 3))
            for i, target_cls in enumerate(self.target_classes):
                if target_cls in model_classes:
                    col_idx = np.where(model_classes == target_cls)[0][0]
                    aligned[:, i] = proba_raw[:, col_idx]
                else:
                    aligned[:, i] = 1e-12
        
        # Normalización defensiva
        aligned = np.clip(aligned, 1e-12, 1.0 - 1e-12)
        aligned /= aligned.sum(axis=1, keepdims=True)
        return aligned
    
    def evaluate(self, X, y_true, verbose=True):
        """Evaluación completa con métricas alineadas al benchmark"""
        proba = self.predict_proba(X)
        preds = self.predict(X)
        y_true = np.asarray(y_true)
        
        # Métricas
        accuracy = (preds == y_true).mean()
        logloss = self._safe_logloss(y_true, proba)
        brier = self._brier_multiclass(y_true, proba)
        ece = self._calculate_ece(y_true, proba)
        
        metrics = {
            'accuracy': float(accuracy),
            'logloss': float(logloss),
            'brier': float(brier),
            'ece': float(ece),
            'n_samples': len(y_true),
            'class_distribution': {
                'away_win': int((y_true == -1).sum()),
                'draw': int((y_true == 0).sum()),
                'home_win': int((y_true == 1).sum())
            }
        }
        
        if verbose:
            print(f"\n📊 Evaluación GBM Predictor")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   LogLoss:   {logloss:.4f}")
            print(f"   Brier:     {brier:.4f}")
            print(f"   ECE:       {ece:.4f}")
            
        return metrics
    
    @staticmethod
    def _safe_logloss(y_true, y_prob, eps=1e-12):
        y_prob = np.clip(y_prob, eps, 1 - eps)
        y_oh = np.zeros_like(y_prob)
        for i, val in enumerate(y_true):
            idx = np.where(np.array([-1, 0, 1]) == val)[0][0]
            y_oh[i, idx] = 1.0
        return -np.mean(np.sum(y_oh * np.log(y_prob), axis=1))
    
    @staticmethod
    def _brier_multiclass(y_true, y_prob):
        y_oh = np.zeros_like(y_prob)
        for i, val in enumerate(y_true):
            idx = np.where(np.array([-1, 0, 1]) == val)[0][0]
            y_oh[i, idx] = 1.0
        return np.mean(np.sum((y_prob - y_oh) ** 2, axis=1))
    
    @staticmethod
    def _calculate_ece(y_true, y_prob, n_bins=10):
        preds = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        accuracies = (preds == y_true).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                ece += np.abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * prop_in_bin
        return ece
    
    def save(self, path: str):
        """Exporta modelo + metadatos para producción"""
        import joblib
        package = {
            'model': self.model,
            'target_classes': self.target_classes,
            'feature_names': self.feature_names,
            'calibrate': self.calibrate,
            'metrics_history': self.metrics_history,
            'export_date': datetime.now().isoformat()
        }
        joblib.dump(package, path)
        print(f"✅ Modelo guardado: {path}")
    
    @classmethod
    def load(cls, path):
        """Carga modelo exportado"""
        import joblib
        package = joblib.load(path)
        predictor = cls(calibrate=package['calibrate'])
        predictor.model = package['model']
        predictor.target_classes = package['target_classes']
        predictor.feature_names = package['feature_names']
        predictor.metrics_history = package['metrics_history']
        predictor.is_fitted = True
        return predictor
