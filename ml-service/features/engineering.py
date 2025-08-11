import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import math

class FeatureEngine:
    """
    Feature engineering pipeline for fraud detection.
    Extracts behavioral patterns and risk indicators from transaction data.
    """
    
    def __init__(self):
        self.user_profiles = {}
        self.merchant_stats = {}
        
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string handling timezone differences"""
        try:
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'
            
            dt = datetime.fromisoformat(timestamp_str)
            
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
                
            return dt
        except:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', ''))
            return dt.replace(tzinfo=timezone.utc)
        
    def build_user_profile(self, transactions: List[Dict]) -> Dict:
        """
        Build behavioral profile from user's transaction history.
        
        Args:
            transactions: List of user's past transactions
            
        Returns:
            Dict with user's spending patterns, preferences, and habits
        """
        if not transactions:
            return self._default_profile()
            
        df = pd.DataFrame(transactions)
        df['timestamp'] = df['timestamp'].apply(self._parse_timestamp)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Calculate spending patterns
        spending_stats = {
            'avg_amount': df['amount'].mean(),
            'std_amount': df['amount'].std(),
            'median_amount': df['amount'].median(),
            'max_amount': df['amount'].max(),
            'q75_amount': df['amount'].quantile(0.75),
            'q25_amount': df['amount'].quantile(0.25),
        }
        
        # Time-based patterns
        time_patterns = {
            'preferred_hours': df['hour'].value_counts().head(3).index.tolist(),
            'daily_frequency': len(df) / max(1, (df['timestamp'].max() - df['timestamp'].min()).days),
            'weekend_ratio': (df['day_of_week'] >= 5).mean(),
            'night_ratio': ((df['hour'] >= 22) | (df['hour'] <= 6)).mean(),
        }
        
        # Geographic patterns
        geo_patterns = {
            'home_lat': df['location'].apply(lambda x: x['lat']).median(),
            'home_lng': df['location'].apply(lambda x: x['lng']).median(),
            'location_radius': self._calculate_movement_radius(df),
        }
        
        # Merchant patterns
        merchant_patterns = {
            'favorite_merchants': df['merchant'].value_counts().head(5).index.tolist(),
            'merchant_diversity': df['merchant'].nunique(),
        }
        
        # Payment patterns
        payment_patterns = {
            'card_types': df['card_type'].value_counts().to_dict(),
            'transaction_types': df['transaction_type'].value_counts().to_dict(),
        }
        
        # General stats
        general_stats = {
            'total_transactions': len(df),
            'active_days': df['timestamp'].dt.date.nunique(),
            'last_transaction': df['timestamp'].max().isoformat()
        }
        
        # Combine all patterns
        profile = {}
        profile.update(spending_stats)
        profile.update(time_patterns)
        profile.update(geo_patterns)
        profile.update(merchant_patterns)
        profile.update(payment_patterns)
        profile.update(general_stats)
        
        return profile
    
    def _default_profile(self) -> Dict:
        """Default profile for new users without transaction history"""
        return {
            'avg_amount': 150.0,
            'std_amount': 100.0,
            'median_amount': 100.0,
            'max_amount': 500.0,
            'q75_amount': 200.0,
            'q25_amount': 50.0,
            'preferred_hours': [12, 18, 20],
            'daily_frequency': 2.0,
            'weekend_ratio': 0.3,
            'night_ratio': 0.1,
            'home_lat': -23.5505,
            'home_lng': -46.6333,
            'location_radius': 10.0,
            'favorite_merchants': ['Amazon', 'McDonald\'s'],
            'merchant_diversity': 10,
            'card_types': {'credit': 0.7, 'debit': 0.3},
            'transaction_types': {'purchase': 0.8, 'withdrawal': 0.2},
            'total_transactions': 0,
            'active_days': 1,
            'last_transaction': datetime.now(timezone.utc).isoformat()
        }
    
    def _calculate_movement_radius(self, df: pd.DataFrame) -> float:
        """Calculate typical movement radius for user"""
        if len(df) < 2:
            return 10.0
            
        locations = df['location'].apply(pd.Series)
        center_lat = locations['lat'].median()
        center_lng = locations['lng'].median()
        
        distances = locations.apply(
            lambda row: self._haversine_distance(
                center_lat, center_lng, row['lat'], row['lng']
            ), axis=1
        )
        
        return distances.quantile(0.9)
    
    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two geographic points using Haversine formula"""
        R = 6371  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def extract_features(self, transaction: Dict, user_history: List[Dict] = None) -> Dict:
        """
        Extract comprehensive feature set from a single transaction.
        
        Args:
            transaction: Current transaction data
            user_history: Optional user transaction history
            
        Returns:
            Dict with extracted features for ML model
        """
        user_id = transaction['user_id']
        
        if user_history:
            profile = self.build_user_profile(user_history)
            self.user_profiles[user_id] = profile
        else:
            profile = self.user_profiles.get(user_id, self._default_profile())
        
        timestamp = self._parse_timestamp(transaction['timestamp'])
        features = {}
        
        # Basic transaction features
        features.update(self._extract_basic_features(transaction, timestamp))
        
        # Amount-based features
        features.update(self._extract_amount_features(transaction, profile))
        
        # Time-based features
        features.update(self._extract_time_features(timestamp, profile))
        
        # Location-based features
        features.update(self._extract_location_features(transaction, profile))
        
        # Merchant-based features
        features.update(self._extract_merchant_features(transaction, profile))
        
        # Velocity features
        features.update(self._extract_velocity_features(timestamp, profile))
        
        # Behavioral features
        features.update(self._extract_behavioral_features(transaction, profile))
        
        # Composite risk features
        features.update(self._extract_composite_features(features))
        
        return features
    
    def _extract_basic_features(self, transaction: Dict, timestamp: datetime) -> Dict:
        """Extract basic transaction features"""
        return {
            'amount': transaction['amount'],
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': int(timestamp.weekday() >= 5),
            'is_night': int(timestamp.hour >= 22 or timestamp.hour <= 6),
            'month': timestamp.month,
            'is_holiday_season': int(timestamp.month in [11, 12, 1]),
        }
    
    def _extract_amount_features(self, transaction: Dict, profile: Dict) -> Dict:
        """Extract amount-related risk features"""
        amount = transaction['amount']
        
        return {
            'amount_vs_avg': amount / max(profile['avg_amount'], 1),
            'amount_vs_median': amount / max(profile['median_amount'], 1),
            'amount_z_score': (amount - profile['avg_amount']) / max(profile['std_amount'], 1),
            'amount_percentile': self._get_amount_percentile(amount, profile),
            'is_round_amount': int(amount % 10 == 0 or amount % 100 == 0),
            'amount_above_max': int(amount > profile['max_amount']),
            'amount_above_q75': int(amount > profile['q75_amount']),
        }
    
    def _extract_time_features(self, timestamp: datetime, profile: Dict) -> Dict:
        """Extract time-based behavioral features"""
        return {
            'is_preferred_hour': int(timestamp.hour in profile['preferred_hours']),
            'hour_deviation': min([abs(timestamp.hour - h) for h in profile['preferred_hours']]),
            'is_unusual_time': int(
                timestamp.hour not in profile['preferred_hours'] and 
                (timestamp.hour >= 22 or timestamp.hour <= 6)
            ),
            'weekend_mismatch': int(
                (timestamp.weekday() >= 5) != (profile['weekend_ratio'] > 0.5)
            ),
        }
    
    def _extract_location_features(self, transaction: Dict, profile: Dict) -> Dict:
        """Extract location-based risk features"""
        current_lat = transaction['location']['lat']
        current_lng = transaction['location']['lng']
        
        distance_from_home = self._haversine_distance(
            profile['home_lat'], profile['home_lng'],
            current_lat, current_lng
        )
        
        return {
            'distance_from_home': distance_from_home,
            'outside_usual_radius': int(distance_from_home > profile['location_radius']),
            'very_far_from_home': int(distance_from_home > profile['location_radius'] * 3),
            'location_risk_score': min(distance_from_home / max(profile['location_radius'], 1), 10),
        }
    
    def _extract_merchant_features(self, transaction: Dict, profile: Dict) -> Dict:
        """Extract merchant-related features"""
        merchant = transaction['merchant']
        
        return {
            'is_favorite_merchant': int(merchant in profile['favorite_merchants']),
            'is_new_merchant': int(merchant not in profile['favorite_merchants']),
            'merchant_risk_category': self._categorize_merchant_risk(merchant),
        }
    
    def _extract_velocity_features(self, timestamp: datetime, profile: Dict) -> Dict:
        """Extract transaction velocity features"""
        last_transaction_time = self._parse_timestamp(profile['last_transaction'])
        time_since_last = (timestamp - last_transaction_time).total_seconds() / 3600
        
        return {
            'time_since_last_transaction': max(0, time_since_last),
            'transactions_last_hour': np.random.poisson(0.5),  # Simulated
            'transactions_last_day': np.random.poisson(profile['daily_frequency']),
            'velocity_risk': int(time_since_last < 0.1 and time_since_last >= 0),
        }
    
    def _extract_behavioral_features(self, transaction: Dict, profile: Dict) -> Dict:
        """Extract behavioral pattern features"""
        return {
            'card_type_mismatch': int(
                transaction['card_type'] not in profile['card_types'] or
                profile['card_types'].get(transaction['card_type'], 0) < 0.1
            ),
            'transaction_type_mismatch': int(
                transaction['transaction_type'] not in profile['transaction_types'] or
                profile['transaction_types'].get(transaction['transaction_type'], 0) < 0.1
            ),
            'user_experience': profile['total_transactions'],
            'user_activity_level': profile['active_days'],
        }
    
    def _extract_composite_features(self, features: Dict) -> Dict:
        """Create composite risk indicators"""
        return {
            'risk_score_composite': self._calculate_composite_risk(features),
            'anomaly_indicators': sum([
                features['amount_above_max'],
                features['outside_usual_radius'],
                features['is_unusual_time'],
                features['velocity_risk'],
                features['card_type_mismatch']
            ]),
        }
    
    def _get_amount_percentile(self, amount: float, profile: Dict) -> float:
        """Calculate amount percentile for user"""
        q25 = profile['q25_amount']
        q75 = profile['q75_amount']
        
        if amount <= q25:
            return 25.0
        elif amount >= q75:
            return 75.0
        else:
            return 25.0 + (amount - q25) / (q75 - q25) * 50.0
    
    def _categorize_merchant_risk(self, merchant: str) -> int:
        """Categorize merchant risk level (0=low, 1=medium, 2=high)"""
        high_risk_keywords = ['crypto', 'gambling', 'casino', 'adult', 'pharmacy']
        medium_risk_keywords = ['gas', 'atm', 'cash', 'transfer']
        
        merchant_lower = merchant.lower()
        
        for keyword in high_risk_keywords:
            if keyword in merchant_lower:
                return 2
                
        for keyword in medium_risk_keywords:
            if keyword in merchant_lower:
                return 1
                
        return 0
    
    def _calculate_composite_risk(self, features: Dict) -> float:
        """Calculate weighted composite risk score"""
        risk_components = [
            features['amount_z_score'] * 0.3,
            features['location_risk_score'] * 0.2,
            features['velocity_risk'] * 0.2,
            features['is_unusual_time'] * 0.1,
            features['merchant_risk_category'] * 0.1,
            features['card_type_mismatch'] * 0.1
        ]
        
        return sum(risk_components)
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names in expected order"""
        return [
            # Basic features
            'amount', 'hour', 'day_of_week', 'is_weekend', 'is_night', 'month', 'is_holiday_season',
            # Amount features
            'amount_vs_avg', 'amount_vs_median', 'amount_z_score', 'amount_percentile', 
            'is_round_amount', 'amount_above_max', 'amount_above_q75',
            # Time features
            'is_preferred_hour', 'hour_deviation', 'is_unusual_time', 'weekend_mismatch',
            # Location features
            'distance_from_home', 'outside_usual_radius', 'very_far_from_home', 'location_risk_score',
            # Merchant features
            'is_favorite_merchant', 'is_new_merchant', 'merchant_risk_category',
            # Velocity features
            'time_since_last_transaction', 'transactions_last_hour', 'transactions_last_day', 'velocity_risk',
            # Behavioral features
            'card_type_mismatch', 'transaction_type_mismatch', 'user_experience', 'user_activity_level',
            # Composite features
            'risk_score_composite', 'anomaly_indicators'
        ]