import pandas as pd
import logging
from typing import Tuple, Any
from models.ensemble import FraudDetectionEnsemble
from data.generator import TransactionGenerator
from features.engineering import FeatureEngine
from config import Config

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handle model training and feature preparation"""
    
    def __init__(self):
        self.ensemble = FraudDetectionEnsemble()
        self.generator = TransactionGenerator()
        self.feature_engine = FeatureEngine()
    
    def train_model(self) -> dict:
        """Train the fraud detection ensemble model"""
        logger.info("Starting model training process...")
        
        # Generate synthetic training data
        training_data = self.generator.generate_training_data()
        
        # Extract features
        X, feature_names = self._prepare_training_features(training_data)
        y = training_data['is_fraud'].values
        
        # Train ensemble model
        metrics = self.ensemble.train(X, y, feature_names)
        
        # Save trained model
        self.ensemble.save_model()
        
        logger.info("Model training completed and saved")
        return metrics
    
    def _prepare_training_features(self, df: pd.DataFrame) -> Tuple[Any, list]:
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
            features = self.feature_engine.extract_features(
                transaction.to_dict(), 
                user_history if len(user_history) > 0 else None
            )
            
            all_features.append(features)
        
        # Convert to structured format
        features_df = pd.DataFrame(all_features)
        feature_names = self.feature_engine.get_feature_names()
        
        # Ensure all features are present
        for name in feature_names:
            if name not in features_df.columns:
                features_df[name] = 0
        
        return features_df[feature_names].values, feature_names