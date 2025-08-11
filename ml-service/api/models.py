from pydantic import BaseModel
from typing import List, Dict

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