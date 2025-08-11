import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime, timedelta
from config import Config

logger = logging.getLogger(__name__)

class TransactionGenerator:
    """Generate synthetic transaction data for model training"""
    
    def __init__(self):
        self.merchants = {
            'low_risk': ["McDonald's", "Subway", "Starbucks", "Supermarket", "Gas Station"],
            'medium_risk': ["Online Store", "ATM", "Money Transfer", "Uber", "Food Delivery"],
            'high_risk': ["Crypto Exchange", "Online Casino", "Cash Advance", "Suspicious Vendor"]
        }
        
        self.locations = {
            "City Center": {"lat": -23.5505, "lng": -46.6333, "risk": 0.02},
            "Suburb": {"lat": -23.6000, "lng": -46.7000, "risk": 0.01},
            "Airport": {"lat": -23.4356, "lng": -46.4731, "risk": 0.05},
            "Border Town": {"lat": -25.0000, "lng": -55.0000, "risk": 0.15}
        }
        
        self.card_types = ["credit", "debit", "prepaid"]
        self.transaction_types = ["purchase", "withdrawal", "transfer", "payment"]
    
    def generate_training_data(self, n_samples: int = None) -> pd.DataFrame:
        """Generate synthetic transaction data for model training"""
        if n_samples is None:
            n_samples = Config.TRAINING_SAMPLES
            
        logger.info(f"Generating {n_samples} synthetic transactions...")
        
        np.random.seed(Config.RANDOM_STATE)
        random.seed(Config.RANDOM_STATE)
        
        user_profiles = self._create_user_profiles()
        transactions = []
        
        for i in range(n_samples):
            user_id = f"user_{random.randint(1, Config.N_USERS)}"
            profile = user_profiles[user_id]
            
            is_fraud = random.random() < profile['fraud_probability']
            
            if is_fraud:
                transaction_data = self._generate_fraud_transaction(profile)
            else:
                transaction_data = self._generate_normal_transaction(profile)
            
            # Create timestamp
            days_ago = random.randint(0, Config.DAYS_BACK)
            timestamp = datetime.now() - timedelta(
                days=days_ago, 
                hours=random.randint(0, 23), 
                minutes=random.randint(0, 59)
            )
            timestamp = timestamp.replace(hour=transaction_data['hour'])
            
            # Set location coordinates
            base_location = self.locations[transaction_data['location_name']]
            location = {
                "lat": base_location["lat"] + random.uniform(-0.02, 0.02),
                "lng": base_location["lng"] + random.uniform(-0.02, 0.02)
            }
            
            transaction = {
                "id": f"txn_{i+1:06d}",
                "user_id": user_id,
                "amount": round(transaction_data['amount'], 2),
                "merchant": transaction_data['merchant'],
                "location": location,
                "timestamp": timestamp.isoformat(),
                "card_type": transaction_data['card_type'],
                "transaction_type": random.choice(self.transaction_types),
                "is_fraud": int(is_fraud)
            }
            
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        fraud_count = df['is_fraud'].sum()
        
        logger.info(f"Generated {len(df)} transactions: {fraud_count} frauds ({fraud_count/len(df)*100:.1f}%)")
        
        return df
    
    def _create_user_profiles(self) -> dict:
        """Create user profiles with different risk levels"""
        user_profiles = {}
        
        for user_id in range(1, Config.N_USERS + 1):
            risk_category = random.choice(['low_risk', 'medium_risk', 'high_risk'])
            
            if risk_category == 'low_risk':
                profile = {
                    'home_location': 'City Center',
                    'avg_amount': random.uniform(50, 200),
                    'active_hours': random.sample(range(8, 20), 3),
                    'preferred_merchants': random.sample(self.merchants['low_risk'], 3),
                    'fraud_probability': Config.FRAUD_RATE_LOW_RISK
                }
            elif risk_category == 'medium_risk':
                profile = {
                    'home_location': random.choice(['City Center', 'Suburb', 'Airport']),
                    'avg_amount': random.uniform(100, 500),
                    'active_hours': random.sample(range(6, 23), 4),
                    'preferred_merchants': random.sample(
                        self.merchants['low_risk'] + self.merchants['medium_risk'], 3
                    ),
                    'fraud_probability': Config.FRAUD_RATE_MEDIUM_RISK
                }
            else:  # high_risk
                profile = {
                    'home_location': 'Border Town',
                    'avg_amount': random.uniform(200, 1000),
                    'active_hours': random.sample(range(0, 24), 5),
                    'preferred_merchants': random.sample(
                        self.merchants['medium_risk'] + self.merchants['high_risk'], 2
                    ),
                    'fraud_probability': Config.FRAUD_RATE_HIGH_RISK
                }
            
            user_profiles[f"user_{user_id}"] = profile
        
        return user_profiles
    
    def _generate_fraud_transaction(self, profile: dict) -> dict:
        """Generate fraudulent transaction patterns"""
        if random.random() < 0.7:  # Obvious fraud
            return {
                'amount': random.uniform(2000, 10000),
                'hour': random.choice([2, 3, 4, 23, 0, 1]),
                'merchant': random.choice(self.merchants['high_risk']),
                'location_name': 'Border Town',
                'card_type': random.choice(['debit', 'prepaid'])
            }
        else:  # Subtle fraud
            return {
                'amount': random.uniform(500, 2000),
                'hour': random.choice(profile['active_hours'] + [22, 23, 0, 1]),
                'merchant': random.choice(self.merchants['medium_risk'] + self.merchants['high_risk']),
                'location_name': profile['home_location'] if random.random() < 0.4 else random.choice(list(self.locations.keys())),
                'card_type': random.choice(self.card_types)
            }
    
    def _generate_normal_transaction(self, profile: dict) -> dict:
        """Generate normal transaction patterns"""
        amount = max(5, np.random.normal(profile['avg_amount'], profile['avg_amount'] * 0.3))
        hour = random.choice(profile['active_hours']) if random.random() < 0.8 else random.randint(8, 22)
        merchant = random.choice(profile['preferred_merchants']) if random.random() < 0.7 else random.choice(self.merchants['low_risk'])
        location_name = profile['home_location'] if random.random() < 0.9 else 'City Center'
        card_type = "credit" if random.random() < 0.8 else "debit"
        
        return {
            'amount': amount,
            'hour': hour,
            'merchant': merchant,
            'location_name': location_name,
            'card_type': card_type
        }