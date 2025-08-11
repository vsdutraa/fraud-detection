from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import random
import logging
import time

# Add local modules to path
sys.path.append('features')
sys.path.append('models')

from engineering import FeatureEngine
from ensemble import FraudDetectionEnsemble

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection service using ensemble machine learning",
    version="1.0.0"
)

# Global instances
ensemble_model = FraudDetectionEnsemble()
feature_engine = FeatureEngine()
performance_metrics = []

class Transaction(BaseModel):
    """Transaction data model for API requests"""
    id: str
    user_id: str
    amount: float
    merchant: str
    location: Dict[str, float]  # {"lat": float, "lng": float}
    timestamp: str
    card_type: str = "credit"
    transaction_type: str = "purchase"

class FraudPrediction(BaseModel):
    """Fraud prediction response model"""
    transaction_id: str
    ensemble_score: float
    rf_score: float
    isolation_score: float
    risk_level: str
    model_agreement: str
    confidence: str
    risk_factors: List[str]
    processing_time_ms: int
    timestamp: str
    feature_count: int

def generate_training_data(n_samples: int = 15000) -> pd.DataFrame:
    """
    Generate synthetic transaction data for model training.
    Creates realistic patterns for both normal and fraudulent transactions.
    """
    logger.info(f"Generating {n_samples} synthetic transactions...")
    
    np.random.seed(42)
    random.seed(42)
    
    # Define merchant categories by risk level
    merchants = {
        'low_risk': ["McDonald's", "Subway", "Starbucks", "Supermarket", "Gas Station"],
        'medium_risk': ["Online Store", "ATM", "Money Transfer", "Uber", "Food Delivery"],
        'high_risk': ["Crypto Exchange", "Online Casino", "Cash Advance", "Suspicious Vendor"]
    }
    
    # Geographic locations with associated risk levels
    locations = {
        "City Center": {"lat": -23.5505, "lng": -46.6333, "risk": 0.02},
        "Suburb": {"lat": -23.6000, "lng": -46.7000, "risk": 0.01},
        "Airport": {"lat": -23.4356, "lng": -46.4731, "risk": 0.05},
        "Border Town": {"lat": -25.0000, "lng": -55.0000, "risk": 0.15}
    }
    
    # Create user profiles with different risk levels
    user_profiles = {}
    for user_id in range(1, 1001):
        risk_category = random.choice(['low_risk', 'medium_risk', 'high_risk'])
        
        if risk_category == 'low_risk':
            profile = {
                'home_location': 'City Center',
                'avg_amount': random.uniform(50, 200),
                'active_hours': random.sample(range(8, 20), 3),
                'preferred_merchants': random.sample(merchants['low_risk'], 3),
                'fraud_probability': 0.02
            }
        elif risk_category == 'medium_risk':
            profile = {
                'home_location': random.choice(['City Center', 'Suburb', 'Airport']),
                'avg_amount': random.uniform(100, 500),
                'active_hours': random.sample(range(6, 23), 4),
                'preferred_merchants': random.sample(merchants['low_risk'] + merchants['medium_risk'], 3),
                'fraud_probability': 0.05
            }
        else:  # high_risk
            profile = {
                'home_location': 'Border Town',
                'avg_amount': random.uniform(200, 1000),
                'active_hours': random.sample(range(0, 24), 5),
                'preferred_merchants': random.sample(merchants['medium_risk'] + merchants['high_risk'], 2),
                'fraud_probability': 0.15
            }
        
        user_profiles[f"user_{user_id}"] = profile
    
    # Generate transactions
    transactions = []
    for i in range(n_samples):
        user_id = f"user_{random.randint(1, 1000)}"
        profile = user_profiles[user_id]
        
        # Determine if transaction is fraudulent
        is_fraud = random.random() < profile['fraud_probability']
        
        if is_fraud:
            # Generate fraudulent transaction patterns
            if random.random() < 0.7:  # Obvious fraud
                amount = random.uniform(2000, 10000)
                hour = random.choice([2, 3, 4, 23, 0, 1])
                merchant = random.choice(merchants['high_risk'])
                location_name = 'Border Town'
                card_type = random.choice(['debit', 'prepaid'])
            else:  # Subtle fraud
                amount = random.uniform(500, 2000)
                hour = random.choice(profile['active_hours'] + [22, 23, 0, 1])
                merchant = random.choice(merchants['medium_risk'] + merchants['high_risk'])
                location_name = profile['home_location'] if random.random() < 0.4 else random.choice(list(locations.keys()))
                card_type = random.choice(['credit', 'debit'])
        else:
            # Generate normal transaction patterns
            amount = max(5, np.random.normal(profile['avg_amount'], profile['avg_amount'] * 0.3))
            hour = random.choice(profile['active_hours']) if random.random() < 0.8 else random.randint(8, 22)
            merchant = random.choice(profile['preferred_merchants']) if random.random() < 0.7 else random.choice(merchants['low_risk'])
            location_name = profile['home_location'] if random.random() < 0.9 else 'City Center'
            card_type = "credit" if random.random() < 0.8 else "debit"
        
        # Create timestamp
        days_ago = random.randint(0, 90)
        timestamp = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        timestamp = timestamp.replace(hour=hour)
        
        # Set location coordinates
        base_location = locations[location_name]
        location = {
            "lat": base_location["lat"] + random.uniform(-0.02, 0.02),
            "lng": base_location["lng"] + random.uniform(-0.02, 0.02)
        }
        
        transaction = {
            "id": f"txn_{i+1:06d}",
            "user_id": user_id,
            "amount": round(amount, 2),
            "merchant": merchant,
            "location": location,
            "timestamp": timestamp.isoformat(),
            "card_type": card_type,
            "transaction_type": random.choice(["purchase", "withdrawal", "transfer", "payment"]),
            "is_fraud": int(is_fraud)
        }
        
        transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    fraud_count = df['is_fraud'].sum()
    
    logger.info(f"Generated {len(df)} transactions: {fraud_count} frauds ({fraud_count/len(df)*100:.1f}%)")
    
    return df

def prepare_training_features(df: pd.DataFrame):
    """Extract features from transaction dataframe for model training"""
    logger.info("Extracting features from training data...")
    
    all_features = []
    
    for _, transaction in df.iterrows():
        # Get user's transaction history (excluding current transaction)
        user_history = df[
            (df['user_id'] == transaction['user_id']) & 
            (df.index != transaction.name)
        ].to_dict('records')
        
        # Extract features
        features = feature_engine.extract_features(
            transaction.to_dict(), 
            user_history if len(user_history) > 0 else None
        )
        
        all_features.append(features)
    
    # Convert to structured format
    features_df = pd.DataFrame(all_features)
    feature_names = feature_engine.get_feature_names()
    
    # Ensure all features are present
    for name in feature_names:
        if name not in features_df.columns:
            features_df[name] = 0
    
    return features_df[feature_names].values, feature_names

def train_fraud_model():
    """Train the fraud detection ensemble model"""
    logger.info("Starting model training process...")
    
    # Generate synthetic training data
    training_data = generate_training_data(15000)
    
    # Extract features
    X, feature_names = prepare_training_features(training_data)
    y = training_data['is_fraud'].values
    
    # Train ensemble model
    metrics = ensemble_model.train(X, y, feature_names)
    
    # Save trained model
    ensemble_model.save_model()
    
    logger.info("Model training completed and saved")
    return metrics

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting fraud detection service...")
    
    # Try to load existing model, train new one if not found
    if not ensemble_model.load_model():
        logger.info("No existing model found. Training new model...")
        train_fraud_model()
    else:
        logger.info("Existing model loaded successfully")

@app.get("/")
async def root():
    """API health check endpoint"""
    return {
        "service": "Fraud Detection API",
        "status": "running",
        "model_loaded": ensemble_model.is_trained,
        "version": "1.0.0",
        "features": len(ensemble_model.feature_names) if ensemble_model.feature_names else 0
    }

@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud probability for a transaction.
    
    Args:
        transaction: Transaction data
        
    Returns:
        Fraud prediction with risk assessment
    """
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

@app.post("/retrain")
async def retrain_model():
    """Retrain the fraud detection model with new data"""
    try:
        metrics = train_fraud_model()
        return {
            "message": "Model retrained successfully",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/metrics")
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

@app.get("/model-comparison")
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

@app.get("/test-data")
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

@app.get("/feature-importance")
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

@app.get("/health")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)