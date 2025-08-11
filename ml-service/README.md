# ML Service

Fraud detection engine with modular architecture.

## Architecture

```
ml-service/
├── api/                 # REST API layer
├── data/                # Data generation
├── training/            # Model training
├── features/            # Feature engineering
├── models/              # ML ensemble
└── config.py           # Configuration
```

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

## Configuration

All settings centralized in `config.py`:

- Model parameters (RF_N_ESTIMATORS, etc.)
- Risk thresholds (HIGH_RISK_THRESHOLD, etc.)
- Training configuration (TRAINING_SAMPLES, etc.)

## Features

**Amount**: spending vs user average, z-score, percentiles  
**Time**: preferred hours, unusual timing patterns  
**Location**: distance from home, movement radius  
**Merchant**: favorite stores, risk categories  
**Velocity**: transaction frequency, rapid sequences  
**Behavioral**: card type patterns, user experience

## Usage

```python
# Start service
uvicorn main:app --reload --port 8001

# Test prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"id": "test", "user_id": "user_123", ...}' | jq
```

## Dependencies

```bash
pip install -r requirements.txt
```

Contains: fastapi, uvicorn, pandas, scikit-learn, numpy, pydantic
