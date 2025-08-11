class Config:
    """Configuration settings for fraud detection service"""
    
    # API Settings
    API_TITLE = "Fraud Detection API"
    API_DESCRIPTION = "Real-time fraud detection service using ensemble machine learning"
    API_VERSION = "1.0.0"
    API_HOST = "0.0.0.0"
    API_PORT = 8001
    
    # Model Parameters
    RF_N_ESTIMATORS = 300
    RF_MAX_DEPTH = 20
    RF_MIN_SAMPLES_SPLIT = 2
    RF_MIN_SAMPLES_LEAF = 1
    RF_MAX_FEATURES = 'sqrt'
    RF_RANDOM_STATE = 42
    
    ISOLATION_N_ESTIMATORS = 200
    ISOLATION_CONTAMINATION = 0.1
    ISOLATION_MAX_SAMPLES = 'auto'
    ISOLATION_MAX_FEATURES = 1.0
    ISOLATION_RANDOM_STATE = 42
    
    # Ensemble Configuration
    RF_WEIGHT = 0.7
    ISOLATION_WEIGHT = 0.3
    HIGH_RISK_THRESHOLD = 0.8
    MEDIUM_RISK_THRESHOLD = 0.5
    
    # Training Parameters
    TRAINING_SAMPLES = 15000
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Data Generation
    N_USERS = 1000
    DAYS_BACK = 90
    FRAUD_RATE_LOW_RISK = 0.02
    FRAUD_RATE_MEDIUM_RISK = 0.05
    FRAUD_RATE_HIGH_RISK = 0.15
    
    # System Settings
    MAX_PERFORMANCE_METRICS = 100
    SAVED_MODELS_PATH = "saved_models"
    
    # Logging
    LOG_LEVEL = "INFO"