from fastapi import APIRouter, HTTPException
import numpy as np
import time
import logging
from datetime import datetime
from typing import List

from api.models import Transaction, FraudPrediction
from models.ensemble import FraudDetectionEnsemble
from features.engineering import FeatureEngine
from training.trainer import ModelTrainer

logger = logging.getLogger(__name__)

# Global instances
router = APIRouter()
ensemble_model = FraudDetectionEnsemble()
feature_engine = FeatureEngine()
model_trainer = ModelTrainer()
performance_metrics: List[int] = []

@router.get("/")
async def root():
    """API health check endpoint"""
    return {
        "service": "Fraud Detection API",
        "status": "running",
        "model_loaded": ensemble_model.is_trained,
        "version": "1.0.0",
        "features": len(ensemble_model.feature_names) if ensemble_model.feature_names else 0
    }

@router.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: Transaction):
    """Predict fraud probability for a transaction"""
    start_time = time.time()
    
    if not ensemble_model.is_trained:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Extract features from transaction
        features = feature_engine.extract_features(transaction.dict())
        
        # Prepare feature array for model
        feature_array = []
        for name in ensemble_model.feature_names:
            feature_array.append(features.get(name, 0))
        
        feature_array = np.array([feature_array])
        
        # Get model prediction
        prediction = ensemble_model.predict(feature_array)
        
        # Identify specific risk factors
        risk_factors = _analyze_risk_factors(features)
        
        # Add model-specific explanations
        model_explanations = ensemble_model.explain_prediction(prediction, features)
        all_risk_factors = risk_factors + model_explanations
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        performance_metrics.append(processing_time)
        
        # Keep only recent performance metrics
        if len(performance_metrics) > 100:
            performance_metrics.pop(0)
        
        # Build response
        result = FraudPrediction(
            transaction_id=transaction.id,
            ensemble_score=prediction["ensemble_score"],
            rf_score=prediction["rf_score"],
            isolation_score=prediction["isolation_score"],
            risk_level=prediction["risk_level"],
            model_agreement=prediction["model_agreement"],
            confidence=prediction["confidence"],
            risk_factors=all_risk_factors[:5],  # Limit to top 5 factors
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat(),
            feature_count=len(ensemble_model.feature_names)
        )
        
        logger.info(f"Prediction: {transaction.id} -> {prediction['risk_level']} "
                   f"(score: {prediction['ensemble_score']:.3f}, {processing_time}ms)")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/retrain")
async def retrain_model():
    """Retrain the fraud detection model with new data"""
    try:
        metrics = model_trainer.train_model()
        return {
            "message": "Model retrained successfully",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@router.get("/metrics")
async def get_model_metrics():
    """Get detailed model performance metrics"""
    avg_processing_time = sum(performance_metrics) / len(performance_metrics) if performance_metrics else 0
    
    return {
        "model_performance": ensemble_model.model_metrics,
        "system_performance": {
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "total_predictions": len(performance_metrics),
            "model_status": "trained" if ensemble_model.is_trained else "not_trained",
            "feature_count": len(ensemble_model.feature_names),
            "ensemble_weights": {
                "random_forest": ensemble_model.rf_weight,
                "isolation_forest": ensemble_model.isolation_weight
            }
        },
        "feature_analysis": ensemble_model.get_feature_importance_analysis(),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/model-comparison")
async def compare_model_performance():
    """Compare performance of individual models vs ensemble"""
    if not ensemble_model.is_trained:
        return {"error": "Model not trained"}
    
    metrics = ensemble_model.model_metrics
    
    return {
        "random_forest": {
            "precision": metrics["rf_precision"],
            "recall": metrics["rf_recall"],
            "f1_score": metrics["rf_f1"],
            "roc_auc": metrics["rf_roc_auc"],
            "oob_score": metrics["rf_oob_score"]
        },
        "isolation_forest": {
            "precision": metrics["isolation_precision"],
            "recall": metrics["isolation_recall"],
            "f1_score": metrics["isolation_f1"]
        },
        "ensemble": {
            "precision": metrics["ensemble_precision"],
            "recall": metrics["ensemble_recall"],
            "f1_score": metrics["ensemble_f1"],
            "roc_auc": metrics["ensemble_roc_auc"]
        },
        "best_performer": {
            "f1_score": max(
                ("Random Forest", metrics["rf_f1"]),
                ("Isolation Forest", metrics["isolation_f1"]),
                ("Ensemble", metrics["ensemble_f1"])
            ),
            "roc_auc": max(
                ("Random Forest", metrics["rf_roc_auc"]),
                ("Ensemble", metrics["ensemble_roc_auc"])
            )
        }
    }

@router.get("/test-data")
async def generate_test_transactions():
    """Generate sample transactions for testing the API"""
    test_cases = []
    
    # Normal transaction
    test_cases.append({
        "id": "test_normal_001",
        "user_id": "user_123",
        "amount": 45.90,
        "merchant": "McDonald's",
        "location": {"lat": -23.5505, "lng": -46.6333},
        "timestamp": datetime.now().replace(hour=12).isoformat(),
        "card_type": "credit",
        "transaction_type": "purchase"
    })
    
    # Moderately suspicious transaction
    test_cases.append({
        "id": "test_moderate_002",
        "user_id": "user_123",
        "amount": 850.00,
        "merchant": "Online Store",
        "location": {"lat": -23.5505, "lng": -46.6333},
        "timestamp": datetime.now().replace(hour=22).isoformat(),
        "card_type": "credit",
        "transaction_type": "purchase"
    })
    
    # Highly suspicious transaction
    test_cases.append({
        "id": "test_suspicious_003",
        "user_id": "user_123",
        "amount": 3500.00,
        "merchant": "Crypto Exchange",
        "location": {"lat": -8.0476, "lng": -34.8770},
        "timestamp": datetime.now().replace(hour=3).isoformat(),
        "card_type": "debit",
        "transaction_type": "transfer"
    })
    
    # Anomalous but low amount
    test_cases.append({
        "id": "test_anomaly_004",
        "user_id": "user_123",
        "amount": 100.00,
        "merchant": "Starbucks",
        "location": {"lat": -12.9714, "lng": -38.5014},
        "timestamp": datetime.now().replace(hour=14).isoformat(),
        "card_type": "prepaid",
        "transaction_type": "withdrawal"
    })
    
    # Extremely suspicious transaction
    test_cases.append({
        "id": "test_fraud_005",
        "user_id": "user_123",
        "amount": 9999.99,
        "merchant": "Suspicious Vendor",
        "location": {"lat": -25.0000, "lng": -55.0000},
        "timestamp": datetime.now().replace(hour=2).isoformat(),
        "card_type": "prepaid",
        "transaction_type": "withdrawal"
    })
    
    return {
        "test_transactions": test_cases,
        "usage": "Use POST /predict with any of these transactions to test the API",
        "expected_risk_levels": {
            "test_normal_001": "LOW - typical transaction pattern",
            "test_moderate_002": "MEDIUM - elevated amount and late hour",
            "test_suspicious_003": "HIGH - high amount, high-risk merchant, unusual location",
            "test_anomaly_004": "MEDIUM - unusual location and payment method",
            "test_fraud_005": "HIGH - multiple high-risk indicators"
        }
    }

@router.get("/feature-importance")
async def get_feature_importance():
    """Get detailed feature importance analysis"""
    if not ensemble_model.is_trained:
        return {"error": "Model not trained"}
    
    analysis = ensemble_model.get_feature_importance_analysis()
    
    # Categorize features by type
    feature_categories = {
        "amount_features": [f for f in ensemble_model.feature_names if 'amount' in f],
        "time_features": [f for f in ensemble_model.feature_names if any(x in f for x in ['hour', 'time', 'day'])],
        "location_features": [f for f in ensemble_model.feature_names if any(x in f for x in ['location', 'distance', 'home'])],
        "merchant_features": [f for f in ensemble_model.feature_names if 'merchant' in f],
        "behavioral_features": [f for f in ensemble_model.feature_names if any(x in f for x in ['card', 'user', 'velocity'])],
        "composite_features": [f for f in ensemble_model.feature_names if any(x in f for x in ['risk', 'anomaly', 'composite'])]
    }
    
    return {
        "importance_analysis": analysis,
        "feature_categories": feature_categories,
        "insights": {
            "most_important": "Features with >5% importance are critical for detection",
            "optimization_target": "Features with <2% importance could be removed for efficiency",
            "top_10_impact": f"{analysis.get('top_10_contribution_pct', 0):.1f}% of decisions come from top 10 features"
        }
    }

@router.get("/health")
async def health_check():
    """Comprehensive health check for the fraud detection service"""
    model_loaded = ensemble_model.random_forest is not None and ensemble_model.isolation_forest is not None
    
    return {
        "status": "healthy" if ensemble_model.is_trained else "degraded",
        "components": {
            "ensemble_model": ensemble_model.is_trained,
            "random_forest": ensemble_model.random_forest is not None,
            "isolation_forest": ensemble_model.isolation_forest is not None,
            "feature_engine": feature_engine is not None
        },
        "performance": {
            "recent_predictions": len(performance_metrics),
            "avg_response_time_ms": sum(performance_metrics[-10:]) / len(performance_metrics[-10:]) if performance_metrics else 0
        },
        "model_info": {
            "features_count": len(ensemble_model.feature_names),
            "last_trained": ensemble_model.model_metrics.get('last_trained', 'unknown')
        },
        "timestamp": datetime.now().isoformat()
    }

def _analyze_risk_factors(features: dict) -> List[str]:
    """Analyze features and identify specific risk factors"""
    risk_factors = []
    
    if features.get('amount_above_max', 0):
        risk_factors.append("Amount exceeds user's historical maximum")
    
    if features.get('very_far_from_home', 0):
        risk_factors.append("Location unusually far from typical area")
    
    if features.get('is_unusual_time', 0):
        risk_factors.append("Transaction time outside normal pattern")
    
    if features.get('merchant_risk_category', 0) >= 2:
        risk_factors.append("High-risk merchant category")
    
    if features.get('velocity_risk', 0):
        risk_factors.append("High transaction velocity detected")
    
    if features.get('anomaly_indicators', 0) >= 3:
        risk_factors.append("Multiple anomaly indicators present")
    
    return risk_factors