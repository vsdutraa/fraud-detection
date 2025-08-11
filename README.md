# Fraud Detection System

Real-time fraud detection using machine learning ensemble.

## What it does

Analyzes financial transactions and returns fraud probability in under 50ms. Uses Random Forest + Isolation Forest to catch both known patterns and new anomalies.

## Structure

```
ml-service/              # ML API service
├── api/                 # REST API endpoints and models
│   ├── endpoints.py     # All API routes
│   └── models.py        # Request/response models
├── data/                # Data generation and processing
│   └── generator.py     # Synthetic transaction generator
├── training/            # Model training logic
│   └── trainer.py       # Ensemble training pipeline
├── features/            # Feature engineering
│   └── engineering.py   # Behavioral pattern extraction
├── models/              # ML models
│   └── ensemble.py      # Random Forest + Isolation Forest
├── config.py           # Centralized configuration
├── main.py             # FastAPI application
└── requirements.txt    # Dependencies

frontend/               # Dashboard (in progress)
api-gateway/            # Auth layer (planned)
```

## Tech Stack

- **Backend**: Python, FastAPI, scikit-learn
- **Models**: Random Forest, Isolation Forest
- **Frontend**: Next.js (coming next)

## Quick Start

```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload --port 8001
```

## API

```bash
# Test prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_001",
    "user_id": "user_123",
    "amount": 1500,
    "merchant": "Online Store",
    "location": {"lat": -23.5505, "lng": -46.6333},
    "timestamp": "2024-01-15T14:30:00Z",
    "card_type": "credit",
    "transaction_type": "purchase"
  }' | jq

# Check metrics
curl http://localhost:8001/metrics | jq
```

## Performance

- **Latency**: ~40ms per prediction
- **Accuracy**: F1-Score 92%
- **Features**: 35 behavioral indicators
- **Models**: RF (70%) + Isolation (30%)

## Development

- [x] Modular ML service with ensemble model
- [x] Feature engineering pipeline
- [x] REST API with monitoring
- [x] Centralized configuration management
- [ ] Frontend dashboard
- [ ] Production deployment
