import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class FraudDetectionEnsemble:
    """
    Ensemble model combining supervised and unsupervised learning approaches.
    Uses Random Forest for pattern recognition and Isolation Forest for anomaly detection.
    """
    
    def __init__(self):
        self.random_forest = None
        self.isolation_forest = None
        self.feature_names = []
        self.is_trained = False
        self.model_metrics = {}
        
        # Model weights for ensemble combination
        self.rf_weight = 0.7
        self.isolation_weight = 0.3
        
        # Risk classification thresholds
        self.high_risk_threshold = 0.8
        self.medium_risk_threshold = 0.5
        
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Train the ensemble model on provided data.
        
        Args:
            X: Feature matrix
            y: Target labels (0=normal, 1=fraud)
            feature_names: List of feature names
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting ensemble training...")
        
        self.feature_names = feature_names
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest for supervised learning
        self._train_random_forest(X_train, y_train)
        
        # Train Isolation Forest for anomaly detection
        self._train_isolation_forest(X_train, y_train)
        
        # Evaluate ensemble performance
        metrics = self._evaluate_models(X_test, y_test)
        
        self.model_metrics = metrics
        self.is_trained = True
        
        logger.info(f"Training complete. Ensemble F1: {metrics['ensemble_f1']:.3f}")
        
        return metrics
    
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Random Forest classifier"""
        logger.info("Training Random Forest...")
        
        self.random_forest = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
        
        self.random_forest.fit(X_train, y_train)
    
    def _train_isolation_forest(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Isolation Forest for anomaly detection"""
        logger.info("Training Isolation Forest...")
        
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=0.1,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            n_jobs=-1
        )
        
        # Train only on normal transactions for better anomaly detection
        normal_transactions = X_train[y_train == 0]
        self.isolation_forest.fit(normal_transactions)
    
    def _evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate individual models and ensemble performance"""
        logger.info("Evaluating models...")
        
        # Random Forest predictions
        rf_predictions = self.random_forest.predict(X_test)
        rf_probabilities = self.random_forest.predict_proba(X_test)[:, 1]
        
        # Isolation Forest predictions
        isolation_scores = self.isolation_forest.decision_function(X_test)
        isolation_normalized = self._normalize_isolation_scores(isolation_scores)
        isolation_predictions = (isolation_normalized > 0.5).astype(int)
        
        # Ensemble predictions
        ensemble_scores = self._combine_predictions(rf_probabilities, isolation_normalized)
        ensemble_predictions = (ensemble_scores > 0.5).astype(int)
        
        # Calculate metrics for each model
        metrics = {
            # Random Forest metrics
            "rf_precision": float(precision_score(y_test, rf_predictions)),
            "rf_recall": float(recall_score(y_test, rf_predictions)),
            "rf_f1": float(f1_score(y_test, rf_predictions)),
            "rf_roc_auc": float(roc_auc_score(y_test, rf_probabilities)),
            "rf_oob_score": float(self.random_forest.oob_score_),
            
            # Isolation Forest metrics
            "isolation_precision": float(precision_score(y_test, isolation_predictions)),
            "isolation_recall": float(recall_score(y_test, isolation_predictions)),
            "isolation_f1": float(f1_score(y_test, isolation_predictions)),
            
            # Ensemble metrics
            "ensemble_precision": float(precision_score(y_test, ensemble_predictions)),
            "ensemble_recall": float(recall_score(y_test, ensemble_predictions)),
            "ensemble_f1": float(f1_score(y_test, ensemble_predictions)),
            "ensemble_roc_auc": float(roc_auc_score(y_test, ensemble_scores)),
            
            # General statistics
            "training_samples": len(X_test) * 4,  # Approximation of training size
            "test_samples": len(X_test),
            "fraud_rate": float(y_test.mean()),
            "feature_count": len(self.feature_names),
            "rf_weight": self.rf_weight,
            "isolation_weight": self.isolation_weight,
            
            # Feature importance from Random Forest
            "feature_importance": list(zip(self.feature_names, self.random_forest.feature_importances_)),
            "top_features": sorted(
                zip(self.feature_names, self.random_forest.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:15],
            
            "last_trained": datetime.now().isoformat()
        }
        
        return metrics
    
    def _normalize_isolation_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize Isolation Forest scores to 0-1 range.
        Isolation Forest returns negative values for anomalies.
        """
        inverted_scores = -scores  # Convert to positive values for anomalies
        
        min_score = inverted_scores.min()
        max_score = inverted_scores.max()
        
        if max_score == min_score:
            return np.zeros_like(inverted_scores)
        
        normalized = (inverted_scores - min_score) / (max_score - min_score)
        return normalized
    
    def _combine_predictions(self, rf_probabilities: np.ndarray, isolation_scores: np.ndarray) -> np.ndarray:
        """Combine predictions from both models using weighted average"""
        return self.rf_weight * rf_probabilities + self.isolation_weight * isolation_scores
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make fraud prediction on new transaction.
        
        Args:
            X: Feature array for single transaction
            
        Returns:
            Dictionary with prediction results and explanations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get predictions from individual models
        rf_probability = self.random_forest.predict_proba(X)[:, 1][0]
        isolation_score = self.isolation_forest.decision_function(X)[0]
        
        # Normalize isolation score
        isolation_normalized = max(0, min(1, (-isolation_score + 0.5) / 1.0))
        
        # Calculate ensemble score
        ensemble_score = self._combine_predictions(
            np.array([rf_probability]), 
            np.array([isolation_normalized])
        )[0]
        
        # Determine risk level
        risk_level = self._classify_risk_level(ensemble_score)
        
        # Calculate model agreement and confidence
        agreement = self._calculate_model_agreement(rf_probability, isolation_normalized)
        confidence = self._calculate_prediction_confidence(ensemble_score, rf_probability, isolation_normalized)
        
        return {
            "ensemble_score": float(ensemble_score),
            "rf_score": float(rf_probability),
            "isolation_score": float(isolation_score),
            "isolation_score_normalized": float(isolation_normalized),
            "risk_level": risk_level,
            "model_agreement": agreement,
            "confidence": confidence
        }
    
    def _classify_risk_level(self, score: float) -> str:
        """Classify risk level based on ensemble score"""
        if score > self.high_risk_threshold:
            return "HIGH"
        elif score > self.medium_risk_threshold:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_model_agreement(self, rf_score: float, isolation_score: float) -> str:
        """Check if both models agree on the prediction"""
        rf_prediction = "fraud" if rf_score > 0.5 else "normal"
        isolation_prediction = "fraud" if isolation_score > 0.5 else "normal"
        
        return "high_agreement" if rf_prediction == isolation_prediction else "disagreement"
    
    def _calculate_prediction_confidence(self, ensemble_score: float, rf_score: float, isolation_score: float) -> str:
        """Calculate confidence level of the prediction"""
        # High confidence if scores are extreme or models agree strongly
        extreme_score = (ensemble_score > 0.9) or (ensemble_score < 0.1)
        both_extreme = (rf_score > 0.8 and isolation_score > 0.8) or (rf_score < 0.2 and isolation_score < 0.2)
        
        if extreme_score and both_extreme:
            return "very_high"
        elif extreme_score or both_extreme:
            return "high"
        elif abs(rf_score - isolation_score) < 0.2:
            return "medium"
        else:
            return "low"
    
    def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """Analyze feature importance from Random Forest"""
        if not self.is_trained:
            return {}
        
        importance_data = self.model_metrics["feature_importance"]
        
        # Categorize features by importance level
        high_importance = [(name, score) for name, score in importance_data if score > 0.05]
        medium_importance = [(name, score) for name, score in importance_data if 0.02 < score <= 0.05]
        low_importance = [(name, score) for name, score in importance_data if score <= 0.02]
        
        return {
            "high_importance_features": high_importance,
            "medium_importance_features": medium_importance,
            "low_importance_features": low_importance,
            "total_features": len(importance_data),
            "top_10_contribution_pct": sum([score for _, score in importance_data[:10]]) * 100
        }
    
    def save_model(self, base_path: str = "saved_models"):
        """Save trained ensemble to disk"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save individual models
        joblib.dump(self.random_forest, f"{base_path}/random_forest.pkl")
        joblib.dump(self.isolation_forest, f"{base_path}/isolation_forest.pkl")
        joblib.dump(self.feature_names, f"{base_path}/feature_names.pkl")
        
        # Save configuration
        config = {
            "rf_weight": self.rf_weight,
            "isolation_weight": self.isolation_weight,
            "high_risk_threshold": self.high_risk_threshold,
            "medium_risk_threshold": self.medium_risk_threshold,
            "is_trained": self.is_trained
        }
        
        with open(f"{base_path}/ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        with open(f"{base_path}/model_metrics.json", "w") as f:
            json.dump(self.model_metrics, f, indent=2)
        
        logger.info("Model saved successfully")
    
    def load_model(self, base_path: str = "saved_models"):
        """Load trained ensemble from disk"""
        try:
            self.random_forest = joblib.load(f"{base_path}/random_forest.pkl")
            self.isolation_forest = joblib.load(f"{base_path}/isolation_forest.pkl")
            self.feature_names = joblib.load(f"{base_path}/feature_names.pkl")
            
            with open(f"{base_path}/ensemble_config.json", "r") as f:
                config = json.load(f)
                self.rf_weight = config["rf_weight"]
                self.isolation_weight = config["isolation_weight"]
                self.high_risk_threshold = config["high_risk_threshold"]
                self.medium_risk_threshold = config["medium_risk_threshold"]
                self.is_trained = config["is_trained"]
            
            with open(f"{base_path}/model_metrics.json", "r") as f:
                self.model_metrics = json.load(f)
            
            logger.info("Model loaded successfully")
            return True
            
        except FileNotFoundError:
            logger.warning("Model files not found")
            return False
    
    def explain_prediction(self, prediction_result: Dict, feature_values: Dict) -> List[str]:
        """Generate human-readable explanations for predictions"""
        explanations = []
        
        rf_score = prediction_result["rf_score"]
        isolation_score = prediction_result["isolation_score_normalized"]
        agreement = prediction_result["model_agreement"]
        confidence = prediction_result["confidence"]
        
        # Model agreement explanation
        if agreement == "high_agreement":
            explanations.append(f"Both models agree (RF: {rf_score:.2f}, Anomaly: {isolation_score:.2f})")
        else:
            explanations.append(f"Models disagree - RF: {rf_score:.2f}, Anomaly: {isolation_score:.2f}")
        
        # Confidence explanation
        explanations.append(f"Prediction confidence: {confidence}")
        
        # Feature-based explanations
        important_features = [name for name, _ in self.model_metrics["top_features"][:5]]
        
        for feature in important_features:
            if feature in feature_values and feature_values[feature] > 1:
                explanations.append(f"'{feature}' indicates elevated risk ({feature_values[feature]:.2f})")
        
        return explanations