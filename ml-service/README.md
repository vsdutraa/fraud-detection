# ML Service

Fraud detection engine with ensemble learning.

## Models

- **Random Forest**: Pattern recognition (70% weight)
- **Isolation Forest**: Anomaly detection (30% weight)
- **Feature Engine**: 35 behavioral features

## Performance

- ~40ms latency
- F1-Score 92%
- 35 features extracted per transaction

## Endpoints

```bash
POST /predict          # Fraud prediction
GET /metrics           # Model performance
GET /model-comparison  # RF vs Isolation vs Ensemble
GET /test-data         # Sample transactions
GET /health            # Service status
```

## Features

**Amount**: spending vs user average, z-score, percentiles  
**Time**: preferred hours, unusual timing patterns  
**Location**: distance from home, movement radius  
**Merchant**: favorite stores, risk categories  
**Velocity**: transaction frequency, rapid sequences  
**Behavioral**: card type patterns, user experience

## Usage

```python
# Basic prediction
transaction = {
    "id": "txn_001",
    "user_id": "user_123",
    "amount": 1500.00,
    "merchant": "Crypto Exchange",
    "location": {"lat": -23.5505, "lng": -46.6333},
    "timestamp": "2024-01-15T03:30:00Z",
    "card_type": "debit",
    "transaction_type": "transfer"
}

# Expected response
{
  "ensemble_score": 0.89,
  "rf_score": 0.95,
  "isolation_score": -0.18,
  "risk_level": "HIGH",
  "model_agreement": "high_agreement",
  "processing_time_ms": 43
}
```

## Dependencies

```
fastapi
uvicorn
pandas
scikit-learn
numpy
```
